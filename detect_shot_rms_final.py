import pyaudio
import struct
import math
from statistics import mean
from datetime import datetime
import pandas as pd
import numpy as np
import keyboard
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


p = pyaudio.PyAudio()

RATE = 44100
CHANNELS = 1
INPUT_BLOCK_TIME = 0.10
SECOND_TEST_AMBIENT = 5
FORMAT = pyaudio.paInt16
SHORT_NORMALIZE = 0.1 / 32768.0
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)


def find_input_device():  # search you device for the index location of a mic
    device_index = None
    for i in range(p.get_device_count()):
        devinfo = p.get_device_info_by_index(i)
        print("Device %d: %s" % (i, devinfo["name"]))

        for keyword in ["mic", "input"]:
            if keyword in devinfo["name"].lower():
                print("Found an input: device %d - %s" % (i, devinfo["name"]))
                device_index = i
                return device_index

    if device_index == None:
        print("No preferred input found; using default input device.")

    return device_index


def open_mic_stream():
    device_index = find_input_device()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
    )

    return stream


def get_rms(block):
    # RMS amplitude is defined as the square root of the
    # mean over time of the square of the amplitude.
    # so we need to convert this string of bytes into
    # a string of 16-bit samples...

    # we will get one short out for each
    # two chars in the string.
    count = len(block) / 2
    format = "%dh" % (count)
    shorts = struct.unpack(format, block)

    # iterate over the block.
    sum_squares = 0.0
    for sample in shorts:
        # sample is a signed short in +/- 32768.
        # normalize it to 1.0
        n = sample * SHORT_NORMALIZE
        sum_squares += n * n

    return math.sqrt((sum_squares / count))


def detect_shots_dB_Level(rms_list, num_of_shots):
    rms_list.sort()
    ambient_dB = rms_list[-num_of_shots:]
    reduction_value = 0.9
    return float(min(ambient_dB)) * reduction_value


def count_down():
    print(f"Establishing Threshold Level for Ambient Sound in 3 Second")
    time.sleep(1)
    print(f"3...")
    time.sleep(1)
    print(f"2...")
    time.sleep(1)
    print(f"1...")
    time.sleep(1)
    print(f"Begin Firing")
    
def count_down_none_ambient():
    print(f"Begin Firing in 3 seconds:")
    time.sleep(1)
    print(f"3...")
    time.sleep(1)
    print(f"2...")
    time.sleep(1)
    print(f"1...")
    time.sleep(1)
    print(f"Begin Firing")


def ambient_rms(ambient_time_test, stream):
    ambient_rms_list = []
    print(f"Testing Ambient RMS Value")
    for i in range(int(SECOND_TEST_AMBIENT / INPUT_BLOCK_TIME)):
        ambient_block = stream.read(
            INPUT_FRAMES_PER_BLOCK
        )  # record audio for ambient sound value
        ambient_rms = get_rms(ambient_block)
        ambient_rms_list.append(ambient_rms)
    print(f"Min RMS Value for Ambient Noise is: {min(ambient_rms_list)}")
    print(f"Max RMS Value for Ambient Noise is: {max(ambient_rms_list)}")
    print(f"Mean RMS Value for Ambient Noise is: {mean(ambient_rms_list)}")
    return np.quantile(ambient_rms_list, 0.92)


def num_shot_ambient_rms(stream):
    ambient_rms_list = []
    num_of_shots = int(input("How many shots will be fired to test ambient sound: "))
    time_to_shot_for_ambient = int(
        input(f"How many second to record {num_of_shots} Shots: ")
    )
    count_down()
    for i in range(int(time_to_shot_for_ambient / INPUT_BLOCK_TIME)):
        ambient_block = stream.read(
            INPUT_FRAMES_PER_BLOCK
        )  # record audio for ambient sound value
        ambient_rms = get_rms(ambient_block)
        ambient_rms_list.append(ambient_rms)
    third_quant = np.quantile(ambient_rms_list, 0.75)
    new_shot_dB_level_detect = detect_shots_dB_Level(ambient_rms_list, num_of_shots)
    print(
        f"Number of shots recorded {num_of_shots}, dB level for labeling set to {new_shot_dB_level_detect}"
    )
    return new_shot_dB_level_detect


def shot_timestamps(max_ambient, stream, weapon_label, event_label):
  
    rsm_list_ploting = []
    time_stamp_list = []
    shot_ref_list = []
    weapon_label_list = []
    event_label_list = []
    
    while True:  # making a loop
        if keyboard.is_pressed(
            " "
        ):  # if key space is pressed.You can also use right,left,up,down and others like a,b,c,etc.
            print("Exiting Audio Recording and Shot TimeStamps")
            break  # finishing the loop
        else:
            ambient_block = stream.read(
                INPUT_FRAMES_PER_BLOCK
            )  # record audio for ambient sound value
            ambient_rms, time_stamp_now = get_rms(ambient_block), datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            if ambient_rms > max_ambient:
                rsm_list_ploting.append(ambient_rms)
                time_stamp_list.append(time_stamp_now)
                shot_ref_list.append(1)
                weapon_label_list.append(weapon_label)
                event_label_list.append(event_label)
                print(
                    f'Shot, RMS Value {ambient_rms}, TimeStamp {datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}'
                )
            else:
                shot_ref_list.append(0)
                rsm_list_ploting.append(ambient_rms)
                time_stamp_list.append(time_stamp_now)
                weapon_label_list.append(weapon_label)
                event_label_list.append(event_label)
                pass

    return rsm_list_ploting, time_stamp_list, shot_ref_list, weapon_label_list, event_label_list
  
def shot_timestamps_none_ambient(none_ambient_required_value, stream):
  
    rsm_list_ploting = []
    time_stamp_list = []
    shot_ref_list = []
    
    while True:  # making a loop
        if keyboard.is_pressed(
            " "
        ):  # if key space is pressed.You can also use right,left,up,down and others like a,b,c,etc.
            print("Exiting Audio Recording and Shot TimeStamps")
            time.sleep(3)
            break  # finishing the loop
        else:
            ambient_block = stream.read(
                INPUT_FRAMES_PER_BLOCK
            )  # record audio for ambient sound value
            ambient_rms, time_stamp_now = get_rms(ambient_block), datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            if ambient_rms > none_ambient_required_value:
                rsm_list_ploting.append(ambient_rms)
                time_stamp_list.append(time_stamp_now)
                shot_ref_list.append(1)
                print(
                    f'Shot, RMS Value {ambient_rms}, TimeStamp {datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")}'
                )
            else:
                shot_ref_list.append(0)
                rsm_list_ploting.append(ambient_rms)
                time_stamp_list.append(time_stamp_now)
                pass
    return rsm_list_ploting, time_stamp_list, shot_ref_list


def get_final_labels(rsm_list_values, intial_rsm_height=0.01, step=0.001):
    confirmed_shot_count = int(input("How many rounds were fired during this iteration?: "))
    print(f'The Number of shots to confirm is {confirmed_shot_count}')
    peaks, _ = find_peaks(rsm_list_values, height=intial_rsm_height)
    current_rsm_height = intial_rsm_height
    exit_count = 0

    while (len(peaks) != confirmed_shot_count):
        if len(peaks) > confirmed_shot_count:
            peaks, _ = find_peaks(rsm_list_values, height=current_rsm_height)
            # print(len(peaks))
            current_rsm_height += step
            # print(intial_rsm_height)
        elif len(peaks) < confirmed_shot_count:
            current_rsm_height = intial_rsm_height
            step = (step * (10 ** -1))
            peaks, _ = find_peaks(rsm_list_values, height=current_rsm_height)

    new_shot_label_df = [0] * len(rsm_list_values)
    indexes = list(peaks)
    print(indexes)
    replacements = [1] * len(peaks)
    print(replacements)

    for i in range(len(indexes)):
        new_shot_label_df[indexes[i]] = replacements[i]

    return peaks, new_shot_label_df


def main(ambient_test):
  
  device_index = find_input_device()

  stream = p.open(
      format=FORMAT,
      channels=CHANNELS,
      rate=RATE,
      input=True,
      input_device_index=1,
      frames_per_buffer=INPUT_FRAMES_PER_BLOCK,
  )
  
  if ambient_test == 1:

    max_ambient = num_shot_ambient_rms(stream)
    
    print(f'Storing ambient treshold: {max_ambient}')

    np.savetxt('ambient_threshold.txt', [max_ambient])
    
    weapon_label = input("Weapon Type (1 = M4A1, 2 = Glock): ")
    event_label = input("Event (1 =  Standing, 2 = Kneeing/Prone, 3 = Walking, 4 = Run/Stop): ")

    (
        rsm_list_ploting_values,
        time_stamp_list_values,
        shot_ref_list_values,
        weapon_label_values, 
        event_label_values
    ) = shot_timestamps(max_ambient, stream, weapon_label, event_label)

    # New code for confirming that the correct number of shots were labeled
    peaks, confirmed_shot_labels = get_final_labels(rsm_list_ploting_values, intial_rsm_height = 0.000006, step = 0.00000025)
    none_ambient_required_value = np.loadtxt('ambient_threshold.txt')

    fig, axs = plt.subplots(2, 1, figsize=(25, 20), gridspec_kw={'hspace': 0.35})
    axs[0].plot(rsm_list_ploting_values, label="RSM of Decibels")
    axs[0].axhline(y=none_ambient_required_value, color="r", linestyle="-", label="Ambient Line")
    axs[0].set_ylabel("RSM Value")
    axs[0].set_xlabel("Time")
    axs[0].set_title("Shot Label Based on Ambient Noise")
    axs[0].grid(True)
    axs[0].legend(title = "Legend")

    axs[1].plot(rsm_list_ploting_values, label = "RSM Values")
    axs[1].plot(peaks, np.array(rsm_list_ploting_values)[peaks.astype(int)], marker="*", markersize=10, c="red",
                linestyle=" ", label="Confirmed Labels")
    axs[1].set_ylabel("RMS Value")
    axs[1].set_xlabel("Time")
    axs[1].set_title("Shot Label Based on Confirmed Number of Shots")
    axs[1].legend(title="Legend")


    plot_title_str = f'check_plots/confirm_Plot_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.png'
    plt.savefig(plot_title_str)

    plt.show()

    time_stamp_shot_df = {
        "Shot_Label": shot_ref_list_values,
        "Time_Stamps": time_stamp_list_values,
        "RSM_Value": rsm_list_ploting_values,
        "Weapon":weapon_label_values,
        "Event":event_label_values
    }

    confirmed_time_stamp_shot_df = {
        "Shot_Label": confirmed_shot_labels,
        "Time_Stamps": time_stamp_list_values,
        "RSM_Value": rsm_list_ploting_values,
        "Weapon":weapon_label_values,
        "Event":event_label_values
    }

    # Original Code for Finding and Storing Data
    time_stamp_shot_df = pd.DataFrame(time_stamp_shot_df)
    print(time_stamp_shot_df["Shot_Label"].value_counts())
    seperate_file_name = "Shot_Label_Time_Stamp_RG37_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv"
    path = "shot_time_data/" + seperate_file_name
    time_stamp_shot_df.to_csv(path, index=False, header=True)
    running_file = "shot_time_data/Full_Shot_Label_Time_Stamp_RG37.csv"
    time_stamp_shot_df.to_csv(running_file, mode='a', header=False, index=False)

    # Confirmed Shot Code for Finding and Storing Data
    confirmed_time_stamp_shot_df = pd.DataFrame(confirmed_time_stamp_shot_df)
    print(confirmed_time_stamp_shot_df["Shot_Label"].value_counts())
    seperate_file_name = "Confirmed_Shot_Label_Time_Stamp_RG37_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv"
    path = "post_confirm_shot_data/" + seperate_file_name
    confirmed_time_stamp_shot_df.to_csv(path, index=False, header=True)
    running_file = "post_confirm_shot_data/Full_Post_Confirmed__Shot_Label_Time_Stamp_RG37.csv"
    confirmed_time_stamp_shot_df.to_csv(running_file, mode='a', header=False, index=False)


  else: 
    weapon_label = input("Weapon Type (1 = M4A1, 2 = Glock): ")
    event_label = input("Event (1 =  Standing, 2 = Prone, 3 = Walking, 4 = Run/Stop): ")
    none_ambient_required_value =  np.loadtxt('ambient_threshold.txt')
    print(f'Setting saved threshold of: {none_ambient_required_value}')
    
    count_down_none_ambient()
    
    (
        rsm_list_ploting_values,
        time_stamp_list_values,
        shot_ref_list_values,
        weapon_label_values, 
        event_label_values
    ) = shot_timestamps(none_ambient_required_value, stream, weapon_label, event_label)

    # New code for confirming that the correct number of shots were labeled
    peaks, confirmed_shot_labels = get_final_labels(rsm_list_ploting_values, intial_rsm_height=0.000006,
                                                    step=0.00000025)
    none_ambient_required_value = np.loadtxt('ambient_threshold.txt')

    fig, axs = plt.subplots(2, 1, figsize=(25, 20), gridspec_kw={'hspace': 0.35})
    axs[0].plot(rsm_list_ploting_values, label="RSM of Decibels")
    axs[0].axhline(y=none_ambient_required_value, color="r", linestyle="-", label="Ambient Line")
    axs[0].set_ylabel("RSM Value")
    axs[0].set_xlabel("Time")
    axs[0].set_title("Shot Label Based on Ambient Noise")
    axs[0].grid(True)
    axs[0].legend(title="Legend")

    axs[1].plot(rsm_list_ploting_values, label="RSM Values")
    axs[1].plot(peaks, np.array(rsm_list_ploting_values)[peaks.astype(int)], marker="*", markersize=10, c="red",
                linestyle=" ", label="Confirmed Labels")
    axs[1].set_ylabel("RMS Value")
    axs[1].set_xlabel("Time")
    axs[1].set_title("Shot Label Based on Confirmed Number of Shots")
    axs[1].legend(title="Legend")

    plot_title_str = f'check_plots/confirm_Plot_{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.png'
    plt.savefig(plot_title_str)

    plt.show()

    time_stamp_shot_df = {
        "Shot_Label": shot_ref_list_values,
        "Time_Stamps": time_stamp_list_values,
        "RSM_Value": rsm_list_ploting_values,
        "Weapon": weapon_label_values,
        "Event": event_label_values
    }

    confirmed_time_stamp_shot_df = {
        "Shot_Label": confirmed_shot_labels,
        "Time_Stamps": time_stamp_list_values,
        "RSM_Value": rsm_list_ploting_values,
        "Weapon": weapon_label_values,
        "Event": event_label_values
    }

    # Original Code for Finding and Storing Data
    time_stamp_shot_df = pd.DataFrame(time_stamp_shot_df)
    print(time_stamp_shot_df["Shot_Label"].value_counts())
    seperate_file_name = "Shot_Label_Time_Stamp_RG37_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv"
    path = "shot_time_data/" + seperate_file_name
    time_stamp_shot_df.to_csv(path, index=False, header=True)
    running_file = "shot_time_data/Full_Shot_Label_Time_Stamp_RG37.csv"
    time_stamp_shot_df.to_csv(running_file, mode='a', header=False, index=False)

    # Confirmed Shot Code for Finding and Storing Data
    confirmed_time_stamp_shot_df = pd.DataFrame(confirmed_time_stamp_shot_df)
    print(confirmed_time_stamp_shot_df["Shot_Label"].value_counts())
    seperate_file_name = "Confirmed_Shot_Label_Time_Stamp_RG37_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv"
    path = "post_confirm_shot_data/" + seperate_file_name
    confirmed_time_stamp_shot_df.to_csv(path, index=False, header=True)
    running_file = "post_confirm_shot_data/Full_Post_Confirmed__Shot_Label_Time_Stamp_RG37.csv"
    confirmed_time_stamp_shot_df.to_csv(running_file, mode='a', header=False, index=False)
    

if __name__ == "__main__":
    appropriate_yes_response = ["y","Y"]
    appropriate_no_response = ["n","N"]
    ambient_ques_ans = input("Do this iteration require an ambient noise threshold test? (y/n): ")
    if ambient_ques_ans in appropriate_yes_response:
        ambient_test = 1
    elif ambient_ques_ans in appropriate_no_response:
        ambient_test = 0
    else:
        ambient_test = input("Please use responce (y/n): ")
    main(ambient_test)
