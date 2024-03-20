import torch
from prettytable import PrettyTable

def count_parameters(ckpt,form_dict=False):
    if not form_dict:
        x = torch.load(ckpt,map_location='cpu')
    #print(x.keys())
        x = x['network_weights']
    else:
        x = ckpt
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    decoder_params = 0
    for name, parameter in x.items():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
        if 'decoder' in name:
            decoder_params += params
    total_params = total_params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f"Total Decoder Params: {decoder_params}")
    return total_params


if __name__ == '__main__':
    import argparse,os,glob
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',type=str)
    args = parser.parse_args()
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = glob.glob(os.path.join(args.path,'*.pth'))
    for ckpt in files:
        print(f'----------------------{ckpt}--------------------')
        count_parameters(ckpt)
        print('------------------------------------------')