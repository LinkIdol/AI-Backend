import argparse
import torch
import torchvision
import os.path

from model import Generator

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size',         type = int,   default = 16,
	help = 'input batch size')
parser.add_argument('--image_size',         type = int,   default = 128,
	help = 'image size')
parser.add_argument('--code_size',          type = int,   default = 256,
	help = 'length of code')
parser.add_argument('--gen_feature',        type = int,   default = [64, 128, 256, 512, 1024, 2048],    nargs = '+',
	help = 'number of features')
parser.add_argument('--gen_block',          type = int,   default = [2, 2, 2, 2, 2, 1],    nargs = '+',
	help = 'number of blocksr')
parser.add_argument('--model_path',                       default = './model',
	help = 'path to generator')
parser.add_argument('--output_path',                      default = './sample.png',
	help = 'path to output image')
parser.add_argument('--sample_size',        type = int,   default = [10, 10], nargs = 2,
	help = 'size of sample grid')
parser.add_argument('--device',                           default = 'cuda:0',
	help = 'device')

opt = parser.parse_args()

device = torch.device(opt.device)


def Generate():
    gen = Generator(opt.image_size, opt.image_size, opt.gen_feature, opt.gen_block, opt.code_size)
    gen.load_state_dict(torch.load(os.path.join(opt.model_path, 'generator.pt')))
    gen.to(device)

    eigenvalues = torch.load(os.path.join(opt.model_path, 'eigenvalues.pt')).to(device)
    eigenvectors = torch.load(os.path.join(opt.model_path, 'eigenvectors.pt')).to(device)
    code_stdv = eigenvectors.mul(eigenvalues.unsqueeze(0)).pow(2).sum(1).sqrt().to(device)

    noise = torch.randn(opt.sample_size[0] * opt.sample_size[1], opt.code_size).to(device)
    code = torch.matmul(noise.mul(eigenvalues.unsqueeze(0)), eigenvectors.transpose(0, 1)).div(code_stdv.unsqueeze(0))

    generated = torch.cat([gen(code[i * opt.batch_size : (i + 1) * opt.batch_size]) for i in range((code.size(0) - 1) // opt.batch_size + 1)], dim = 0)
    torchvision.utils.save_image(generated, opt.output_path, opt.sample_size[1])

