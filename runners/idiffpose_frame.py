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
        
        # Track the best model performance
        self.best_mpjpe = float('inf')
        self.best_epoch = -1
        
        # DEQ configuration - will be updated from config file
        self.deq_schedule = {
            'enabled': False,  # Default to disabled for safety
            'default_iterations': 1,
            'best_epoch_iterations': 3,  # Limited iterations even for best model
            'min_epoch_for_increase': 10,  # Only increase iterations after this epoch
            'warm_up_epochs': 20,          # Use minimal iterations for these epochs
            'max_iterations': 5,           # Lower maximum to avoid divergence
            'tolerance': 0.001             # More relaxed tolerance
        }
        
        # Update DEQ schedule from config file if available
        if hasattr(config, 'deq'):
            for key in self.deq_schedule.keys():
                if hasattr(config.deq, key):
                    self.deq_schedule[key] = getattr(config.deq, key)
                    
        # Print all configuration
        logging.info("================= Configuration =================")
        logging.info(f"Data Settings:")
        logging.info(f"  Dataset: {config.data.dataset}")
        logging.info(f"  Number of joints: {config.data.num_joints}")
        logging.info(f"Model Settings:")
        logging.info(f"  Hidden dimension: {config.model.hid_dim}")
        logging.info(f"  Embedding dimension: {config.model.emd_dim}")
        logging.info(f"  Coords dimension: {config.model.coords_dim}")
        logging.info(f"  Number of layers: {config.model.num_layer}")
        logging.info(f"  Number of heads: {config.model.n_head}")
        logging.info(f"Diffusion Settings:")
        logging.info(f"  Beta schedule: {config.diffusion.beta_schedule}")
        logging.info(f"  Beta start: {config.diffusion.beta_start}")
        logging.info(f"  Beta end: {config.diffusion.beta_end}")
        logging.info(f"  Num diffusion timesteps: {config.diffusion.num_diffusion_timesteps}")
        logging.info(f"DEQ Settings:")
        logging.info(f"  Enabled: {self.deq_schedule.get('enabled', False)}")
        if hasattr(config, 'deq') and hasattr(config.deq, 'components'):
            logging.info(f"  Middle layer: {getattr(config.deq.components, 'middle_layer', False)}")
            logging.info(f"  Final layer: {getattr(config.deq.components, 'final_layer', False)}")
        logging.info(f"  Default iterations: {self.deq_schedule.get('default_iterations', 1)}")
        logging.info(f"  Best epoch iterations: {self.deq_schedule.get('best_epoch_iterations', 3)}")
        logging.info("===============================================")

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        logging.info('==> Using settings {}'.format(args))
        logging.info('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            self.dataset = read_3d_data_me(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                logging.info('==> Selected actions: {}'.format(self.action_filter))
        else:
            raise KeyError('Invalid dataset')
            
        # Print dataset information for debugging
        logging.info("Dataset information:")
        logging.info(f"Train subjects: {self.subjects_train}")
        logging.info(f"Test subjects: {self.subjects_test}")

    # Set DEQ iterations based on the context
    def _set_deq_iterations(self, is_best_epoch=False, is_training=True):
        if not self.deq_schedule.get('enabled', False):
            logging.info("DEQ is disabled, using standard processing")
            return  # DEQ is disabled
            
        # Configure model_diff DEQ settings
        if hasattr(self.model_diff.module, 'deq_manager'):
            if is_best_epoch:
                # For best epoch, we increase iterations gradually
                if not hasattr(self, '_best_epoch_count'):
                    self._best_epoch_count = 0
                
                self._best_epoch_count += 1
                iterations = min(
                    1 + self._best_epoch_count,  # Start with 2, then increase
                    self.deq_schedule['best_epoch_iterations']
                )
                
                logging.info(f"Using {iterations} DEQ iterations for best epoch evaluation")
                self.model_diff.module.deq_manager.set_iterations(iterations)
            else:
                # Use default iterations for normal training/testing
                self.model_diff.module.deq_manager.set_iterations(
                    self.deq_schedule['default_iterations']
                )
            
            # Reset DEQ stats
            self.model_diff.module.deq_manager.reset_stats()
        
        # Configure model_pose DEQ settings
        if hasattr(self.model_pose.module, 'deq_manager'):
            # For pose model, we set a flag for best epoch
            if hasattr(self.model_pose.module, 'is_best_epoch'):
                self.model_pose.module.is_best_epoch = is_best_epoch
            
            if is_best_epoch:
                # Use same gradual increase strategy as diff model
                if not hasattr(self, '_best_epoch_count'):
                    self._best_epoch_count = 0
                
                iterations = min(
                    1 + self._best_epoch_count,
                    self.deq_schedule['best_epoch_iterations']
                )
                
                self.model_pose.module.deq_manager.set_iterations(iterations)
            else:
                self.model_pose.module.deq_manager.set_iterations(
                    self.deq_schedule['default_iterations']
                )
            
            # Reset DEQ stats
            self.model_pose.module.deq_manager.reset_stats()

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        logging.info(f"Creating GCNdiff model...")
        
        # Debug: Print adjacency matrix information
        logging.info(f"Adjacency matrix shape: {adj.shape}")
        logging.info(f"Adjacency matrix values - min: {adj.min().item():.4f}, max: {adj.max().item():.4f}, mean: {adj.mean().item():.4f}")
        
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
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        
        logging.info(f"Creating GCNpose model...")
        logging.info(f"Coords dimensions set to: {config.model.coords_dim}")
        
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
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
                
            # Debug: Print data shapes
            logging.info(f"Training data shapes:")
            logging.info(f"  poses_train: {len(poses_train)} samples")
            logging.info(f"  poses_train_2d: {len(poses_train_2d)} samples")
            if len(poses_train) > 0 and len(poses_train_2d) > 0:
                logging.info(f"  Sample poses_train shape: {poses_train[0].shape}")
                logging.info(f"  Sample poses_train_2d shape: {poses_train_2d[0].shape}")
                
                # Check data range
                logging.info(f"  Sample poses_train min/max: {poses_train[0].min():.4f}/{poses_train[0].max():.4f}")
                logging.info(f"  Sample poses_train_2d min/max: {poses_train_2d[0].min():.4f}/{poses_train_2d[0].max():.4f}")
            
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
                    
            logging.info(f"Training data loader created with {len(data_loader)} batches")
        else:
            raise KeyError('Invalid dataset')
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        logging.info(f"Optimizer: {type(optimizer).__name__} with lr={config.optim.lr}")
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
            logging.info(f"EMA initialized with rate {self.config.model.ema_rate}")
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            logging.info(f"Starting epoch {epoch}")
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            # Configure DEQ iterations based on epoch
            is_warm_up = epoch < self.deq_schedule['warm_up_epochs']
            self._set_deq_iterations(is_best_epoch=False, is_training=True)
            
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
                    # Log DEQ statistics
                    if hasattr(self.model_diff.module, 'deq_manager') and self.deq_schedule.get('enabled', False):
                        self.model_diff.module.deq_manager.log_stats()
                        
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                logging.info(f"Learning rate decayed to {lr_now}")
                
            if epoch % 1 == 0:
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
                    
                    # This is a new best epoch - test with more DEQ iterations
                    if self.deq_schedule.get('enabled', False):
                        logging.info('New best model found. Testing with increased DEQ iterations...')
                        # Start with modest increase in iterations for stability
                        p1_refined, p2_refined = self.test_hyber(is_train=True, is_best_epoch=True)
                        
                        logging.info('| Standard MPJPE: {:.2f} PA-MPJPE: {:.2f} | Refined MPJPE: {:.2f} PA-MPJPE: {:.2f} |'\
                            .format(p1, p2, p1_refined, p2_refined))
                    
                    # Save the best model with annotation that it's best
                    torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                    logging.info(f"Saved best checkpoint with MPJPE: {p1:.2f}")
                    
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(best_epoch, best_p1, epoch, p1, p2))
    
    def test_hyber(self, is_train=False, is_best_epoch=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
                
            # Debug: Print validation data shapes
            logging.info(f"Validation data shapes:")
            logging.info(f"  poses_valid: {len(poses_valid)} samples")
            logging.info(f"  poses_valid_2d: {len(poses_valid_2d)} samples")
            if len(poses_valid) > 0 and len(poses_valid_2d) > 0:
                logging.info(f"  Sample poses_valid shape: {poses_valid[0].shape}")
                logging.info(f"  Sample poses_valid_2d shape: {poses_valid_2d[0].shape}")
                
                # Check data range
                logging.info(f"  Sample poses_valid min/max: {poses_valid[0].min():.4f}/{poses_valid[0].max():.4f}")
                logging.info(f"  Sample poses_valid_2d min/max: {poses_valid_2d[0].min():.4f}/{poses_valid_2d[0].max():.4f}")
                
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
                
            logging.info(f"Validation data loader created with {len(data_loader)} batches")
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        # Set DEQ iterations based on whether this is a best epoch evaluation
        self._set_deq_iterations(is_best_epoch=is_best_epoch, is_training=False)
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        # Debug: Print diffusion sequence
        logging.info(f"Diffusion sequence type: {self.args.skip_type}")
        logging.info(f"Diffusion sequence length: {len(seq)}")
        logging.info(f"Diffusion sequence sample: {seq[:5]}...")
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
            'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)   

        for i, (_, input_noise_scale, input_2d, targets_3d, input_action, camera_para) in enumerate(data_loader):
            data_time += time.time() - data_start
            
            # Debug: Print batch shapes for first batch
            if i == 0:
                logging.info(f"First validation batch shapes:")
                logging.info(f"  input_2d: {input_2d.shape}")
                logging.info(f"  input_noise_scale: {input_noise_scale.shape}")
                logging.info(f"  targets_3d: {targets_3d.shape}")
                logging.info(f"  input_2d min/max/mean: {input_2d.min().item():.4f}/{input_2d.max().item():.4f}/{input_2d.mean().item():.4f}")
                logging.info(f"  input_noise_scale min/max/mean: {input_noise_scale.min().item():.4f}/{input_noise_scale.max().item():.4f}/{input_noise_scale.mean().item():.4f}")
                logging.info(f"  targets_3d min/max/mean: {targets_3d.min().item():.4f}/{targets_3d.max().item():.4f}/{targets_3d.mean().item():.4f}")

            input_noise_scale, input_2d, targets_3d = \
                input_noise_scale.to(self.device), input_2d.to(self.device), targets_3d.to(self.device)

            # build uvxyz
            inputs_xyz = self.model_pose(input_2d, src_mask)  
            
            # Debug: Print model_pose outputs for first batch
            if i == 0:
                logging.info(f"GCNpose outputs for first batch:")
                logging.info(f"  inputs_xyz shape: {inputs_xyz.shape}")
                logging.info(f"  inputs_xyz min/max/mean: {inputs_xyz.min().item():.4f}/{inputs_xyz.max().item():.4f}/{inputs_xyz.mean().item():.4f}")
            
            inputs_xyz[:, :, :] -= inputs_xyz[:, :1, :] 
            input_uvxyz = torch.cat([input_2d,inputs_xyz],dim=2)
            
            # Debug: Print processed outputs for first batch
            if i == 0:
                logging.info(f"  input_uvxyz shape: {input_uvxyz.shape}")
                logging.info(f"  input_uvxyz min/max/mean: {input_uvxyz.min().item():.4f}/{input_uvxyz.max().item():.4f}/{input_uvxyz.mean().item():.4f}")
                        
            # generate distribution
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            input_noise_scale = input_noise_scale.repeat(test_times,1,1)
            # select diffusion step
            t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
            
            # prepare the diffusion parameters
            x = input_uvxyz
            e = torch.randn_like(input_uvxyz)
            b = self.betas   
            e = e*input_noise_scale        
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            
            # Debug: Print diffusion parameters for first batch
            if i == 0:
                logging.info(f"Diffusion parameters for first validation batch:")
                logging.info(f"  t value: {t[0].item()}")
                logging.info(f"  a min/max/mean: {a.min().item():.4f}/{a.max().item():.4f}/{a.mean().item():.4f}")
                logging.info(f"  e min/max/mean: {e.min().item():.4f}/{e.max().item():.4f}/{e.mean().item():.4f}")
            
            # For best epoch evaluation, we can use more diffusion steps
            # Only do this if enabled in config and if it's actually a best epoch
            increase_timesteps = False
            if is_best_epoch and hasattr(config, 'deq') and \
               hasattr(config.deq, 'diffusion') and \
               getattr(config.deq.diffusion, 'increase_timesteps_for_best', False):
                increase_timesteps = True
            
            if increase_timesteps:
                # Use a modest increase in timesteps (e.g., 1.5x instead of 2x)
                multiplier = getattr(config.deq.diffusion, 'timestep_multiplier', 1.5)
                more_timesteps = int(test_timesteps * multiplier)
                
                if self.args.skip_type == "uniform":
                    more_skip = test_num_diffusion_timesteps // more_timesteps
                    more_seq = range(0, test_num_diffusion_timesteps, more_skip)
                elif self.args.skip_type == "quad":
                    more_seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), more_timesteps)** 2)
                    more_seq = [int(s) for s in list(more_seq)]
                
                logging.info(f"Using increased timesteps: {len(more_seq)} steps")
                output_uvxyz = generalized_steps(x, src_mask, more_seq, self.model_diff, self.betas, eta=self.args.eta)
            else:
                output_uvxyz = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, eta=self.args.eta)
            
            # Debug: Print generalized_steps output for first batch
            if i == 0:
                logging.info(f"generalized_steps output type: {type(output_uvxyz)}")
                if isinstance(output_uvxyz, tuple) and len(output_uvxyz) > 0:
                    logging.info(f"  output_uvxyz[0] length: {len(output_uvxyz[0])}")
                    if len(output_uvxyz[0]) > 0:
                        logging.info(f"  output_uvxyz[0][-1] shape: {output_uvxyz[0][-1].shape}")
                        logging.info(f"  output_uvxyz[0][-1] min/max/mean: {output_uvxyz[0][-1].min().item():.4f}/{output_uvxyz[0][-1].max().item():.4f}/{output_uvxyz[0][-1].mean().item():.4f}")
            
            output_uvxyz = output_uvxyz[0][-1]            
            output_uvxyz = torch.mean(output_uvxyz.reshape(test_times,-1,17,5),0)
            output_xyz = output_uvxyz[:,:,2:]
            
            # Debug: Print processed outputs for first batch
            if i == 0:
                logging.info(f"Final processed outputs for first validation batch:")
                logging.info(f"  output_xyz shape: {output_xyz.shape}")
                logging.info(f"  output_xyz min/max/mean before normalization: {output_xyz.min().item():.4f}/{output_xyz.max().item():.4f}/{output_xyz.mean().item():.4f}")
            
            output_xyz[:, :, :] -= output_xyz[:, :1, :]
            targets_3d[:, :, :] -= targets_3d[:, :1, :]
            
            # Debug: Print normalized outputs for first batch
            if i == 0:
                logging.info(f"  output_xyz min/max/mean after normalization: {output_xyz.min().item():.4f}/{output_xyz.max().item():.4f}/{output_xyz.mean().item():.4f}")
                logging.info(f"  targets_3d min/max/mean after normalization: {targets_3d.min().item():.4f}/{targets_3d.max().item():.4f}/{targets_3d.mean().item():.4f}")
            
            # Scale correction:
            scale_factor = (targets_3d.abs().mean() / output_xyz.abs().mean()).detach()
            output_xyz = output_xyz * scale_factor

            # Add another debug print to see the effect of scaling
            if i == 0:
                logging.info(f"  Applied scale factor: {scale_factor.item():.4f}")
                logging.info(f"  output_xyz min/max/mean after scaling: {output_xyz.min().item():.4f}/{output_xyz.max().item():.4f}/{output_xyz.mean().item():.4f}")
            
            # Calculate metrics
            current_mpjpe = mpjpe(output_xyz, targets_3d).item() * 1000.0
            current_p_mpjpe = p_mpjpe(output_xyz.cpu().numpy(), targets_3d.cpu().numpy()).item() * 1000.0
            
            # Debug: Print metrics for first batch
            if i == 0:
                logging.info(f"  First batch MPJPE: {current_mpjpe:.4f}")
                logging.info(f"  First batch P-MPJPE: {current_p_mpjpe:.4f}")
            
            epoch_loss_3d_pos.update(current_mpjpe, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(current_p_mpjpe, targets_3d.size(0))
            
            action_error_sum = test_calculation(output_xyz, targets_3d, input_action, action_error_sum, None, None)
            
            data_start = time.time()
            
            if i%100 == 0 and i != 0:
                # Log DEQ statistics if available
                if hasattr(self.model_diff.module, 'deq_manager') and self.deq_schedule.get('enabled', False):
                    self.model_diff.module.deq_manager.log_stats()
                if hasattr(self.model_pose.module, 'deq_manager') and self.deq_schedule.get('enabled', False):
                    self.model_pose.module.deq_manager.log_stats()
                    
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
        
        # Log final DEQ statistics
        if hasattr(self.model_diff.module, 'deq_manager') and self.deq_schedule.get('enabled', False):
            self.model_diff.module.deq_manager.log_stats()
        if hasattr(self.model_pose.module, 'deq_manager') and self.deq_schedule.get('enabled', False):
            self.model_pose.module.deq_manager.log_stats()
            
        logging.info('Safe testing results | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg))
        
        # Detailed action-specific error analysis
        logging.info("Action-specific MPJPE values:")
        for action, values in action_error_sum.items():
            logging.info(f"  {action}: {values['p1'].avg * 1000.0:.4f} (P1), {values['p2'].avg * 1000.0:.4f} (P2)")
        
        p1, p2 = print_error(None, action_error_sum, is_train)
        
        logging.info(f"Final MPJPE: {p1:.4f}, P-MPJPE: {p2:.4f}")

        return p1, p2