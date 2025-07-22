from antares_http import antares
import pandas as pd
from datetime import datetime
import time
from apscheduler.schedulers.background import BackgroundScheduler
import os

get_data = False

schedule = {
    # Hour: {"lamp": status, "fan": status, "ac": status, "disp": status}
    0:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    1:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    2:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    3:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    4:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    5:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    6:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    7:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    8:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    9:  {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    10: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    11: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    12: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    13: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    14: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    15: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    16: {"lamp": 1, "fan": 1, "ac": 0, "disp": 1},
    17: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    18: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    19: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    20: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    21: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    22: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
    23: {"lamp": 1, "fan": 1, "ac": 0, "disp": 0},
}

antares.setDebug(False)
antares.setAccessKey('fe5c7a15d8c13220:bfd764392a99a094')

# for key, value in latestData['content'].items():
#     print(f'{key}: {value}')

def job_function():
    print(f"Hello World, the time is {datetime.now()}")

    # current_hour = datetime.now().hour
    # current_statuses = schedule.get(current_hour)

    # lamp2 = current_statuses["lamp"]
    # fan2 = current_statuses["fan"]
    # ac2 = current_statuses["ac"]
    # disp2 = current_statuses["disp"]

    # lamp1 = 1
    # fan1 = 1
    # ac1 = 1
    # disp1 = 1

    # data_to_send = {
    #   "source": "flutter_app",
    #   "control_command": "1",
    #   "manual_control": 1,
    #   "system_active_l1": 1,
    #   "fan_status_l1": fan1,
    #   "lamp_status_l1": lamp1,
    #   "ac_status_l1": ac1,
    #   "disp_status_l1": disp1,
    #   "system_active_l2": 1,
    #   "fan_status_l2": fan2,
    #   "lamp_status_l2": lamp2,
    #   "ac_status_l2": ac2,
    #   "disp_status_l2": disp2
    # }

    # # 2. Send Data to Antares
    # print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sending data: {data_to_send}")
    # response = antares.send(data_to_send, 'TADKT-1', 'PMM')

    # # 3. Print Response (optional)
    # # Antares library might print internally or return status/response content
    # print(f"Antares response: {response}") # Check library's return value

    while(True):
        latestData = antares.get('TADKT-1', 'PMM')
        try:
            if latestData['content']['Voltage']:
                energy_data = {
                    'DateTime': [datetime.strptime(latestData['last_modified_time'], '%Y%m%dT%H%M%S')],
                    'Voltage': [latestData['content']['Voltage']],
                    'Current': [latestData['content']['Current']],
                    'Power': [latestData['content']['Power']],
                    'Energy': [latestData['content']['Energy']],
                    'TotalEnergy': [latestData['content']['TotalEnergy']],
                    # 'DailyEnergy': [latestData['content']['DailyEnergy']],
                    # 'limitEnergy': [latestData['content']['limitEnergy']]
                }
                print('got electrical data')
                print(latestData, "\n")
                break
        except:
            print('no electrical data')
            time.sleep(1)
            continue
    
    df = pd.DataFrame(energy_data)
    df.set_index('DateTime', inplace=True)

    if not os.path.isfile('log.csv'):
        df.to_csv('log.csv', mode='w', header=True)
    else:
        df.to_csv('log.csv', mode='a', header=False)

sched = BackgroundScheduler()
sched.add_job(job_function, 'cron', minute=0)
sched.start()
print("Scheduler started. The job will run at the top of the next hour.")

try:
    while True:
        time.sleep(2)
except (KeyboardInterrupt, SystemExit):
    sched.shutdown()

# job_function()


# variable = datetime.now()

# print(variable)



