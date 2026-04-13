import threading
import sys

from rl_isaaclab.utils.misc import ThreadSafeValue


class KeyboardListener:
    def __init__(self,
                 deploy_state_flag: ThreadSafeValue,
                 calib_tactile_flag: ThreadSafeValue,
                 hand_ip: str):
        self.deploy_state_flag = deploy_state_flag
        self.hand_ip = hand_ip
        self.calib_tactile_flag = calib_tactile_flag
        self.last_deploy_state_flag = self.deploy_state_flag.get()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        '''
        deploy_state_flag:
            0: move home
            1: freeze actions
            2: reset env
            3: run policy
        calib_tactile_flag:
            0: silence
            1: calibrate tactile
        '''
        print("[Keyboard] Controls: e=start, w=freeze/resume, q=home, t=calibrate")
        while not self.stop_event.is_set():
            try:
                key = input().strip().lower()
            except EOFError:
                break
            if key == 'q':
                print('[Keyboard] Moving home.')
                self.deploy_state_flag.set(0)
            elif key == 'w':
                if self.deploy_state_flag.get() != 1:
                    self.last_deploy_state_flag = self.deploy_state_flag.get()
                    print('[Keyboard] Freeze actions.')
                    self.deploy_state_flag.set(1)
                else:
                    print('[Keyboard] Continue actions.')
                    self.deploy_state_flag.set(self.last_deploy_state_flag)
            elif key == 'e':
                print('[Keyboard] Start policy.')
                self.deploy_state_flag.set(2)
            elif key == 't':
                self.calib_tactile_flag.set(1)
                print("[Keyboard] Tactile calibration.")
        print("[Keyboard] Keyboard listener stopped.")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
