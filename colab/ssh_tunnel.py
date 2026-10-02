# Forwards ComfyUI to a relay server (e.g. a Lightsail VM) with a reverse SSH tunnel. Tailscale in userspace
# networking mode on Colab is very slow for large transfers, this runs over plain TCP instead.
# The port is bound to 127.0.0.1 on the relay, so reach it from your machine with:
#   ssh -N -L 8188:127.0.0.1:8188 you@relay   then open http://localhost:8188
#
# Reads the TUNNEL_HOST (user@host) and TUNNEL_KEY (base64 of the private key file, `base64 -w0 key`) Colab secrets.
#   !python /content/ComfyUI/colab/ssh_tunnel.py

import base64
import os
import subprocess

KEY = "/root/.ssh/tunnel_key"
PORT = 8188


def main():
    os.makedirs(os.path.dirname(KEY), exist_ok=True)
    with open(os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(base64.b64decode(os.environ["TUNNEL_KEY"]))

    ssh = ["ssh", "-N", "-i", KEY,
           "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
           "-R", f"127.0.0.1:{PORT}:localhost:{PORT}", os.environ["TUNNEL_HOST"]]
    # Reconnect when the connection drops, the relay frees the old forward once its session times out.
    with open("/content/ssh_tunnel.log", "w") as log:
        subprocess.Popen(["bash", "-c", 'while true; do "$@"; sleep 5; done', "ssh_tunnel", *ssh],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Tunneling port {PORT} to {os.environ['TUNNEL_HOST']}, log in /content/ssh_tunnel.log")


if __name__ == "__main__":
    main()
