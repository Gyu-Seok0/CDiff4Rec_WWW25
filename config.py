import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp_created', help='choose the dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save', "--s", action='store_true', help='test with validation')
    parser.add_argument('--save_path', "--sp", type=str, default='./saved_models/', help='save model path')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')
    parser.add_argument('--round', type=int, default=1, help='record the experiment')

    # params for the model
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=50, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    # params for global attention
    parser.add_argument('--alpha', type=float, default=0.1, help='the importance of the user’s own preference')
    parser.add_argument('--beta', type=float, default=0.8, help='the significance of real user')
    parser.add_argument('--gamma', type=float, default=0.1, help='the significance of psuedo user')

    parser.add_argument('--tau', type=float, default=1.0, help='the temperature for consie similarity')
    parser.add_argument('--topk', type=int, default=10, help='he number of top-K similar interest users')
    parser.add_argument('--topk_fake_user', type=int, default = 20000, help='quantity of pseudo-users')

    parser.add_argument('--using_real', "--ur", action = argparse.BooleanOptionalAction, help = "whether using real users (--ur or --no-ur)")
    parser.add_argument('--using_fake', "--uf", action = argparse.BooleanOptionalAction, help = "whether using psuedo-users (--uf or --no-uf)")
    parser.add_argument('--training_fake', "--tf", action = argparse.BooleanOptionalAction, help = "whether training psuedo-users or not (--tf or --no-tf)")
    parser.add_argument('--r_agg',type = str, default = None, help = "att or avg or sim")
    parser.add_argument('--f_agg',type = str, default = None, help = "att or avg or sim")

    parser.add_argument('--random_seed', "--rs", type=int, default=1, help='random_seed')
    parser.add_argument('--lamda', type=float, default = 0.001, help='weight for pseudo user loss')

    args = parser.parse_args()
    return args