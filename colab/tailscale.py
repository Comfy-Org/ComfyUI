# Joins the Colab machine to your tailnet so ComfyUI is reachable from your other devices.
# Colab has no TUN device or systemd, so tailscaled runs in userspace networking mode
# with in-memory state. Use an ephemeral auth key so old Colab machines drop off the tailnet.
#
# In a Colab cell, with the key stored as the TS_AUTHKEY Colab secret:
#   import os
#   from google.colab import userdata
#   os.environ["TS_AUTHKEY"] = userdata.get("TS_AUTHKEY")
#   !python /content/ComfyUI/colab/tailscale.py

import os
import shutil
import subprocess
import time

HOSTNAME = "comfyui-colab"
SOCKET = "/var/run/tailscale/tailscaled.sock"


def main():
    if shutil.which("tailscaled") is None:
        subprocess.check_call("curl -fsSL https://tailscale.com/install.sh | sh", shell=True)

    os.makedirs(os.path.dirname(SOCKET), exist_ok=True)
    with open("/content/tailscaled.log", "w") as log:
        subprocess.Popen(["tailscaled", "--tun=userspace-networking", "--state=mem:", f"--socket={SOCKET}"],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if os.path.exists(SOCKET):
            break
        time.sleep(1)

    subprocess.check_call(["tailscale", f"--socket={SOCKET}", "up", f"--auth-key={os.environ['TS_AUTHKEY']}", f"--hostname={HOSTNAME}"])
    ip = subprocess.check_output(["tailscale", f"--socket={SOCKET}", "ip", "-4"], text=True).strip()
    print(f"ComfyUI will be at http://{HOSTNAME}:8188 (or http://{ip}:8188)")


if __name__ == "__main__":
    main()
