import logging
import subprocess
import webbrowser
from typing import Union, List


def open_browser(
    url: str, 
    browser_path: Union[str, List[str]] = None, 
    browser_profile: Union[str, List[str]] = None
) -> None:
    """
    Launches a custom browser if specified via CLI args, 
    otherwise falls back to the system default browser.
    """
    if browser_path:
        if isinstance(browser_path, list):
            browser_path = " ".join(browser_path)
        if isinstance(browser_profile, list):
            browser_profile = " ".join(browser_profile)

        cmd = [browser_path]
        if browser_profile:
            cmd.append(f"--profile-directory={browser_profile}")
        cmd.append(url)

        try:
            logging.info(f"Launching custom browser: {browser_path}")
            subprocess.Popen(cmd)
            return
        except OSError as e:
            logging.warning(
                f"Failed to launch custom browser '{browser_path}': {e}. "
                f"Falling back to default browser."
            )

    try:
        if not webbrowser.open(url):
            logging.warning(f"Failed to open default web browser for URL: {url}")
    except webbrowser.Error as e:
        logging.error(f"Failed to open web browser: {e}")
