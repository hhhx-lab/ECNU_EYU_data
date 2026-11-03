"""Run inside the selected container before loading external model code."""
import json
import os
from pathlib import Path
import socket


def main():
    routes=Path('/proc/net/route').read_text().splitlines()[1:]
    assert all(line.split()[1]!='00000000' or line.split()[0]=='lo' for line in routes),'default network route present'
    ipv6=Path('/proc/net/ipv6_route')
    if ipv6.exists():assert all(line.split()[-1]=='lo' for line in ipv6.read_text().splitlines()),'external IPv6 route present'
    assert not Path('/public/home/zqchen/.ssh').exists(),'user SSH directory exposed'
    assert not Path('/reference').exists(),'reference/GT exposed'
    assert not Path('/public/home/zqchen/data').exists(),'source data root exposed'
    assert not any(k in os.environ for k in ['HPC_SSH_PASSWORD','HPC_LOGIN_HOST','ECNU_SERVER_ENV_FILE','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']),'private transport env exposed'
    addresses=socket.if_nameindex()
    assert all(name=='lo' for _,name in addresses),'non-loopback network interface exposed'
    print(json.dumps(dict(network_interfaces=addresses,no_default_route=True,no_home_or_gt=True,credentials_environment_absent=True)))


if __name__=='__main__':main()
