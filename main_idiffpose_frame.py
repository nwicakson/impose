import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from runners.idiffpose_frame import IDiffpose


torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    
    parser.add_argument("--seed", type=int, default=19960903, help="Random seed")
    parser.add_argument("--config", type=str, required=True, 
                        help="Path to the config file")
    parser.add_argument("--exp", type=str, default="exp", 
                        help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True, 
                        help="A string for documentation purpose. "\
                            "Will be the name of the log folder.", )
    parser.add_argument("--verbose", type=str, default="info", 
                        help="Verbose level: info | debug | warning | critical")
    parser.add_argument("--ni", action="store_true",
                        help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    ### Diffformer configuration ####
    #Diffusion process hyperparameters
    parser.add_argument("--skip_type", type=str, default="uniform",
                        help="skip according to (uniform or quad(quadratic))")
    parser.add_argument("--eta", type=float, default=0.0, 
                        help="eta used to control the variances of sigma")
    parser.add_argument("--sequence", action="store_true")
    # Diffusion model parameters
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    # load pretrained model
    parser.add_argument('--model_diff_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--model_pose_path', default=None, type=str,
                        help='the path of pretrain model')
    parser.add_argument('--train', action = 'store_true',
                        help='train or evluate')
    #training hyperparameter
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--lr_gamma', default=0.9, type=float, metavar='N',
                        help='weight decay rate')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--decay', default=60, type=int, metavar='N',
                        help='decay frequency(epoch)')
    #test hyperparameter
    parser.add_argument('--test_times', default=5, type=int, metavar='N',
                    help='the number of test times')
    parser.add_argument('--test_timesteps', default=50, type=int, metavar='N',
                    help='the number of test time steps')
    parser.add_argument('--test_num_diffusion_timesteps', default=500, type=int, metavar='N',
                    help='the number of test times')
    
    # DEQ configuration parameters
    parser.add_argument('--deq_enabled', action='store_true',
                  help='enable DEQ in the model')
    parser.add_argument('--deq_middle_layer', action='store_true',
                    help='enable DEQ for middle layer')
    parser.add_argument('--deq_final_layer', action='store_true',
                    help='enable DEQ for final layer')
    parser.add_argument('--deq_iterations', default=5, type=int,
                      help='default number of DEQ iterations')
    parser.add_argument('--deq_best_iterations', default=50, type=int,
                      help='number of DEQ iterations for best epoch')
    parser.add_argument('--best_epoch', action='store_true',
                      help='run evaluation with best epoch settings (more iterations)')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device
    # update configure file
    new_config.training.batch_size = args.batch_size
    new_config.optim.lr = args.lr
    new_config.optim.lr_gamma = args.lr_gamma
    new_config.optim.decay = args.decay

    # Override DEQ settings
    if not hasattr(new_config, 'deq'):
        new_config.deq = argparse.Namespace()
    
    # Use argument values for DEQ settings
    new_config.deq.enabled = args.deq_enabled
    logging.info(f"DEQ enabled status from args: {args.deq_enabled}")

    if not hasattr(new_config.deq, 'components'):
        new_config.deq.components = argparse.Namespace()

    new_config.deq.components.middle_layer = args.deq_middle_layer and args.deq_enabled
    new_config.deq.components.final_layer = args.deq_final_layer and args.deq_enabled
    new_config.deq.default_iterations = args.deq_iterations  # Use value from args
    new_config.deq.best_epoch_iterations = args.deq_best_iterations  # Use value from args
    
    logging.info(f"DEQ configuration: enabled={new_config.deq.enabled}, " +
             f"default_iterations={new_config.deq.default_iterations}, " +
             f"best_epoch_iterations={new_config.deq.best_epoch_iterations}")

    if args.train:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    # Log the test parameters being used
    logging.info(f"Testing parameters: times={args.test_times}, steps={args.test_timesteps}, diffusion={args.test_num_diffusion_timesteps}")
    logging.info(f"DEQ settings: enabled={args.deq_enabled}, iterations={args.deq_iterations}, best_iterations={args.deq_best_iterations}")
    
    try:
        # Create the runner
        runner = IDiffpose(args, config)
        
        # Load models
        runner.create_diffusion_model(args.model_diff_path)
        runner.create_pose_model(args.model_pose_path)
        
        # Prepare data
        runner.prepare_data()
        
        if args.train:
            runner.train()
        else:
            # Use best_epoch flag to determine evaluation mode
            runner.test_hyber(is_train=False, is_best_epoch=args.best_epoch)
            
    except Exception:
        logging.error(traceback.format_exc())

    return 0

if __name__ == "__main__":
    sys.exit(main())