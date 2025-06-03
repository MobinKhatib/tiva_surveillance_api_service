import win32serviceutil
import win32service
import win32event
import servicemanager
import subprocess
import os

# Get the directory where config.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(BASE_DIR, "service_test.log")

# Default Python path
#DEFAULT_PYTHON_PATH = r"C:\Users\Tiva\AppData\Local\Programs\Python\Python310\python.exe"
#DEFAULT_PYTHON_PATH = os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Python", "Python310", "python.exe")

# Use environment variable or fallback to default
#python = os.environ.get("CUSTOM_PYTHON_PATH", DEFAULT_PYTHON_PATH)

class PythonService(win32serviceutil.ServiceFramework):
    _svc_name_ = "TivaSurveillance"
    _svc_display_name_ = "Tiva Surveillance Service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogInfoMsg("Tiva Surveillance Service is starting...")

        try:
            with open(log_path, "w") as f:
                f.write("SvcDoRun entered.\n")

            working_dir = BASE_DIR
            script = os.path.join(working_dir, "server.py")
            python = r"C:\Users\Tiva\AppData\Local\Programs\Python\Python310\python.exe"
 # This is default, it can change by user
            #python = r"C:\Users\IT-Center\AppData\Local\Programs\Python\Python312\python.exe"
            #python = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')

            #with open(os.path.join(working_dir, "service.log"), "w") as log_file:
            #    log_file.write(f"Launching: {python} {script}\n")

            process = subprocess.Popen(
                [python, script],
                cwd=working_dir,
                #stdout=log_file,
                #stderr=log_file
            )

            win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
            process.terminate()
            

        except Exception as e:
            with open(log_path, "w") as f:
                f.write(f"Service failed: {str(e)}")



if __name__ == '__main__':
    import win32serviceutil
    win32serviceutil.HandleCommandLine(PythonService)
