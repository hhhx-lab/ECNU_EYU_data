import torch
from monai.bundle import ConfigParser
import configs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = ConfigParser(configs.NETWORKS_CONFIG)
parser.parse(True)
vae = parser.get_parsed_content('autoencoder_def').to(device)
chk = torch.load(configs.PATH_NAME_WEIGHTS_VAE, weights_only=True, map_location=device)
vae.load_state_dict(chk)
print('VAE loaded OK, state_dict keys:', len(chk.keys()))