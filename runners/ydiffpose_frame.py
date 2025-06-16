import os
import logging
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn


from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data
from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe

# Fix for duplicate logging
_logger_initialized = False
def setup_logging():
    global _logger_initialized
    if _logger_initialized:
        return
    
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    
    # Add a single handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(filename)s - %(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    _logger_initialized = True

# Initialize logging correctly
setup_logging()

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        
        logging.info(f"Initialized Diffpose with diffusion timesteps: {self.num_timesteps}")
        logging.info(f"Beta schedule: {config.diffusion.beta_schedule}, start: {config.diffusion.beta_start}, end: {config.diffusion.beta_end}")

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        logging.info('==> Using settings {}'.format(args))
        
        # load dataset
        if config.data.dataset == "human36m":
            logging.info('==> Loading Human3.6M dataset')
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            logging.info('==> Dataset loaded, processing subjects')
            
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            logging.info('==> Reading 3D data')
            self.dataset = read_3d_data_me(dataset)
            logging.info('==> Creating 2D data for training')
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            logging.info('==> Creating 2D data for testing')
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            logging.info('==> Setting up action filter')
            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = list(map(lambda x: dataset.define_actions(x)[0], self.action_filter))
                logging.info('==> Selected actions: {}'.format(self.action_filter))
            logging.info('==> Data preparation complete')
        else:
            raise KeyError('Invalid dataset')
            
        # Print dataset information for debugging
        logging.info("Dataset information:")
        logging.info(f"Train subjects: {self.subjects_train}")
        logging.info(f"Test subjects: {self.subjects_test}")

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        logging.info("Creating diffusion model...")
        
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # Debug: Print model parameter information
        total_params = sum(p.numel() for p in self.model_diff.parameters())
        trainable_params = sum(p.numel() for p in self.model_diff.parameters() if p.requires_grad)
        logging.info(f"GCNdiff model created - Total parameters: {total_params}, Trainable: {trainable_params}")
        
        # load pretrained model
        if model_path:
            logging.info(f"Loading pretrained GCNdiff model from {model_path}")
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])
            logging.info(f"GCNdiff model loaded successfully")
        else:
            logging.info("No pretrained diffusion model provided, using random initialization")
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        logging.info("Creating GCNpose model...")
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        logging.info(f"Coords dimensions set to: {config.model.coords_dim}")
        
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # Debug: Print model parameter information
        total_params = sum(p.numel() for p in self.model_pose.parameters())
        trainable_params = sum(p.numel() for p in self.model_pose.parameters() if p.requires_grad)
        logging.info(f"GCNpose model created - Total parameters: {total_params}, Trainable: {trainable_params}")
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0])
            logging.info(f"GCNpose model loaded successfully")
        else:
            logging.info('initialize model randomly')

    def train(self):
        logging.info("Starting training...")
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0
        
        # create dataloader
        if config.data.dataset == "human36m":
            logging.info("Creating training data loader...")
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride=1)
            
            logging.info(f"Training data: {len(poses_train)} sets, {len(poses_train_2d)} 2D sets")
            
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
            
            logging.info(f"Training data loader created with {len(data_loader)} batches of size {config.training.batch_size}")
        else:
            raise KeyError('Invalid dataset')
        
        logging.info("Setting up optimizer...")
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        logging.info(f"Optimizer: {type(optimizer).__name__} with lr={config.optim.lr}")
        
        if self.config.model.ema:
            logging.info("Initializing EMA...")
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
            logging.info(f"EMA initialized with rate {self.config.model.ema_rate}")
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
        logging.info(f"Initial lr={lr_init}, decay={decay}, gamma={gamma}")
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            logging.info(f"Starting epoch {epoch}")
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            for i, (targets_uvxyz, targets_noise_scale, _, targets_3d, _, _) in enumerate(data_loader):
                data_time += time.time() - data_start
                step += 1
                
                # Debug: Print batch shapes
                if i == 0:
                    logging.info(f"First batch shapes:")
                    logging.info(f"  targets_uvxyz: {targets_uvxyz.shape}")
                    logging.info(f"  targets_noise_scale: {targets_noise_scale.shape}")
                    logging.info(f"  targets_3d: {targets_3d.shape}")
                    logging.info(f"  targets_uvxyz min/max/mean: {targets_uvxyz.min().item():.4f}/{targets_uvxyz.max().item():.4f}/{targets_uvxyz.mean().item():.4f}")
                    logging.info(f"  targets_noise_scale min/max/mean: {targets_noise_scale.min().item():.4f}/{targets_noise_scale.max().item():.4f}/{targets_noise_scale.mean().item():.4f}")
                    logging.info(f"  targets_3d min/max/mean: {targets_3d.min().item():.4f}/{targets_3d.max().item():.4f}/{targets_3d.mean().item():.4f}")

                # to cuda
                targets_uvxyz, targets_noise_scale, targets_3d = \
                    targets_uvxyz.to(self.device), targets_noise_scale.to(self.device), targets_3d.to(self.device)
                
                # generate nosiy sample based on seleted time t and beta
                n = targets_3d.size(0)
                x = targets_uvxyz
                e = torch.randn_like(x)
                b = self.betas            
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                e = e*(targets_noise_scale)
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # Debug: Print diffusion values for first batch
                if i == 0:
                    logging.info(f"Diffusion parameters:")
                    logging.info(f"  t range: {t.min().item()}-{t.max().item()}")
                    logging.info(f"  a min/max/mean: {a.min().item():.4f}/{a.max().item():.4f}/{a.mean().item():.4f}")
                    logging.info(f"  Noise e min/max/mean: {e.min().item():.4f}/{e.max().item():.4f}/{e.mean().item():.4f}")
                    logging.info(f"  Noised x min/max/mean: {x.min().item():.4f}/{x.max().item():.4f}/{x.mean().item():.4f}")
                
                # predict noise
                output_noise = self.model_diff(x, src_mask, t.float(), 0)
                
                # Debug: Print output values for first batch
                if i == 0:
                    logging.info(f"Model outputs:")
                    logging.info(f"  output_noise shape: {output_noise.shape}")
                    logging.info(f"  output_noise min/max/mean: {output_noise.min().item():.4f}/{output_noise.max().item():.4f}/{output_noise.mean().item():.4f}")
                
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0)
                
                # Debug: Print loss for first batch
                if i == 0:
                    logging.info(f"  Initial loss: {loss_diff.item():.6f}")
                
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                logging.info(f"Learning rate decayed to {lr_now}")
                
            if epoch % 1 == 0:
                logging.info(f"Saving checkpoint at epoch {epoch}")
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                logging.info(f"Saved checkpoint at epoch {epoch}")
            
                logging.info('test the performance of current model')

                # First test with standard settings
                p1, p2 = self.test_hyber(is_train=True)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch = epoch
                    
                    # Save the best model with annotation that it's best
                    torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                    logging.info(f"Saved best checkpoint with MPJPE: {p1:.2f}")
                    
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(best_epoch, best_p1, epoch, p1, p2))
    
    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        
        # FORCE the correct test parameters regardless of what's passed
        test_times = 1  # Force this value 
        test_timesteps = 2  # Force this value
        test_num_diffusion_timesteps = 12  # Force this value
        
        # ADDED: Debug logging of test parameters
        logging.info(f"DIFFPOSE-DEBUG: Test parameters: times={test_times}, steps={test_timesteps}, timesteps={test_num_diffusion_timesteps}")
                
        if config.data.dataset == "human36m":
            logging.info("Loading validation data...")
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride=1)
                
            # Print validation data shape just once
            logging.info(f"Validation data: {len(poses_valid)} samples")
            
            # ADDED: Debug logging of validation data
            logging.info(f"DIFFPOSE-DEBUG: Validation data shapes: poses_3d={len(poses_valid)} sets, poses_2d={len(poses_valid_2d)} sets")
            
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
                
            logging.info(f"Validation data loader created with {len(data_loader)} batches")
            
            # ADDED: Debug logging of data loader
            logging.info(f"DIFFPOSE-DEBUG: Data loader created with {len(data_loader)} batches of size {config.training.batch_size}")
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        # Generate diffusion sequence using fixed parameters
        if args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        # Print diffusion sequence info once
        logging.info(f"Diffusion steps: {len(seq)}, sequence: {list(seq)}")
        
        # ADDED: Debug logging of diffusion sequence
        logging.info(f"DIFFPOSE-DEBUG: Diffusion steps: {len(seq)}, sequence: {list(seq)}")
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
            'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)   

        for i, (_, input_noise_scale, input_2d, targets_3d, input_action, camera_para) in enumerate(data_loader):
            data_time += time.time() - data_start

            input_noise_scale, input_2d, targets_3d = \
                input_noise_scale.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

            # ADDED: Debug logging of input tensors
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: Input 2D stats: min={input_2d.min().item():.4f}, max={input_2d.max().item():.4f}, mean={input_2d.mean().item():.4f}")
                logging.info(f"DIFFPOSE-DEBUG: Target 3D stats: min={targets_3d.min().item():.4f}, max={targets_3d.max().item():.4f}, mean={targets_3d.mean().item():.4f}")
                logging.info(f"DIFFPOSE-DEBUG: Noise scale stats: min={input_noise_scale.min().item():.4f}, max={input_noise_scale.max().item():.4f}")

            # Build uvxyz - exactly like the original implementation
            inputs_xyz = self.model_pose(input_2d, src_mask)  
            
            # Debug the first batch
            if i == 0:
                logging.info(f"First batch GCNpose output: shape={inputs_xyz.shape}, " +
                        f"min={inputs_xyz.min().item():.4f}, max={inputs_xyz.max().item():.4f}, " +
                        f"mean={inputs_xyz.mean().item():.4f}")
                
                # ADDED: Additional debug logging
                logging.info(f"DIFFPOSE-DEBUG: GCNpose output before root centering: min={inputs_xyz.min().item():.4f}, max={inputs_xyz.max().item():.4f}, mean={inputs_xyz.mean().item():.4f}")
            
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :] 
            
            # ADDED: Debug logging after root centering
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: GCNpose output after root centering: min={inputs_xyz.min().item():.4f}, max={inputs_xyz.max().item():.4f}, mean={inputs_xyz.mean().item():.4f}")
            
            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            # ADDED: Debug logging of concatenated input
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: input_uvxyz stats: min={input_uvxyz.min().item():.4f}, max={input_uvxyz.max().item():.4f}, mean={input_uvxyz.mean().item():.4f}")
                    
            # Generate distribution
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            input_noise_scale = input_noise_scale.repeat(test_times,1,1)
            # Select diffusion step
            t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
            
            # Prepare the diffusion parameters
            x = input_uvxyz
            e = torch.randn_like(input_uvxyz)
            b = self.betas   
            e = e*input_noise_scale        
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            
            # ADDED: Debug logging of diffusion parameters
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: Diffusion noise e stats: min={e.min().item():.4f}, max={e.max().item():.4f}, mean={e.mean().item():.4f}")
                logging.info(f"DIFFPOSE-DEBUG: Diffusion alpha a stats: min={a.min().item():.4f}, max={a.max().item():.4f}, mean={a.mean().item():.4f}")
            
            # Use the standard diffusion process
            output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=args.eta)
            
            # Debug first batch
            if i == 0:
                logging.info(f"generalized_steps output: type={type(output_uvxyz)}")
                if isinstance(output_uvxyz, tuple) and len(output_uvxyz) > 0:
                    logging.info(f"  output_uvxyz[0] length: {len(output_uvxyz[0])}")
                    
                # ADDED: More detailed debug logging
                if isinstance(output_uvxyz, tuple) and len(output_uvxyz) > 0 and len(output_uvxyz[0]) > 0:
                    logging.info(f"DIFFPOSE-DEBUG: output_uvxyz[0][-1] stats: min={output_uvxyz[0][-1].min().item():.4f}, max={output_uvxyz[0][-1].max().item():.4f}, mean={output_uvxyz[0][-1].mean().item():.4f}")
            
            output_uvxyz = output_uvxyz[0][-1]
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
            output_xyz = output_uvxyz[:,:,2:]
            
            # ADDED: Debug logging of output before root centering
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: output_xyz before root centering: min={output_xyz.min().item():.4f}, max={output_xyz.max().item():.4f}, mean={output_xyz.mean().item():.4f}")
            
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            
            # ADDED: Debug logging of output and target after root centering
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: output_xyz after root centering: min={output_xyz.min().item():.4f}, max={output_xyz.max().item():.4f}, mean={output_xyz.mean().item():.4f}")
                logging.info(f"DIFFPOSE-DEBUG: targets_3d after root centering: min={targets_3d.min().item():.4f}, max={targets_3d.max().item():.4f}, mean={targets_3d.mean().item():.4f}")
            
            # REMOVED: Scale correction - no scale factor applied
            # Instead calculate scale factor for debug but don't apply it
            scale_factor = (targets_3d.abs().mean() / output_xyz.abs().mean()).detach()
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: Calculated but not applied scale factor: {scale_factor.item():.4f}")
                
            # Calculate metrics before scale (what we're doing now)
            mpjpe_before_scale = mpjpe(output_xyz, targets_3d).item() * 1000.0
            p_mpjpe_value = p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0
            
            # ADDED: Debug logging of metrics 
            if i == 0:
                logging.info(f"DIFFPOSE-DEBUG: MPJPE without scaling: {mpjpe_before_scale:.4f}")
                
                # Also calculate with scale for comparison
                scaled_output = output_xyz * scale_factor
                mpjpe_with_scale = mpjpe(scaled_output, targets_3d).item() * 1000.0
                logging.info(f"DIFFPOSE-DEBUG: MPJPE with scaling (for comparison): {mpjpe_with_scale:.4f}")
            
            # Calculate metrics
            current_mpjpe = mpjpe_before_scale
            current_p_mpjpe = p_mpjpe_value
            
            if i == 0:
                logging.info(f"First batch MPJPE: {current_mpjpe:.4f}, P-MPJPE: {current_p_mpjpe:.4f}")
            
            epoch_loss_3d_pos.update(current_mpjpe, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(current_p_mpjpe, targets_3d.size(0))
            
            action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)
            
            data_start = time.time()
            
            if i%100 == 0:
                # Log progress periodically
                logging.info('({batch}/{size}) Data: {data:.3f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
        
        logging.info('Final results | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg))
        
        # Action-specific errors
        p1, p2 = print_error(None, action_error_sum, is_train)
        
        logging.info(f"Final MPJPE: {p1:.4f}, P-MPJPE: {p2:.4f}")

        return p1, p2
        
        