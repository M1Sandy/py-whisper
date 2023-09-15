import pathlib
import torch  
import whisper 
import os
import numpy as np
import subprocess
from flask import Flask, request
from config import *
from whisper.utils import *
from libretranslatepy import LibreTranslateAPI
from progress.bar import Bar
import re

app = Flask(__name__)
lt = LibreTranslateAPI(libretranslate)

def prod_audio(video_file, output_file):
    command = "ffmpeg -y -i \"{}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{}\"".format(
        video_file, output_file)
    # print("Command: " + command)
    subprocess.call(command, shell=True)


def prod_subtitle(full_path, subtitle_file, audio_file):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    loaded_model = whisper.load_model(model , device=device)

    if torch.cuda.is_available():
        print("[*] Using: GPU!")
    else:
        print("[*] Using: CPU")

    # model = whisper.load_model("base", device=devices)
    print(
        f"[*] Model is {'multilingual' if loaded_model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in loaded_model.parameters()):,} parameters."
    )

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(loaded_model.device)

    _, probs = loaded_model.detect_language(mel)
    print(f"[*] Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(language="en", without_timestamps=False, fp16 = False)
    result = whisper.decode(loaded_model, mel, options)
    # print(result.text)

    result = loaded_model.transcribe(audio_file,verbose=True, word_timestamps=True)
    writer = get_writer("srt", full_path)
    writer(result, subtitle_file, {"max_line_width":50, "max_line_count":2, "highlight_words":False} )
    # print(result["text"])
    return

def is_file_available(file_name):
    if(os.path.isfile(file_name)):
        return True
    else:
        return False
    

def translate(src_full, dst_full):
    buff = ""
    count = 0 
    try:
        src_file = open(src_full, 'r')
    except Exception as e:
        print("[-] [{}] Could not be opened!".format(src_full))
        return
    total_lines = len(src_file.readlines())
    src_file.seek(0)

    with Bar('Translating',max = total_lines) as bar:
        for line in src_file:
            line_translated = ""
            count += 1
            # if(line == "ï»¿1\n"):
            #     buff = buff + "1\n"
            #     continue
                        # words = re.findall(r'[^\W\d]+' , line)
            # joinedLine = " ".join(words)


            joinedLine = " ".join(re.findall("[a-zA-Z]+", line))

            if(joinedLine != "" ):#and (line[0].isalpha or line[0].isdigit)):
                try:

                    line_translated = lt.translate(line.lower(), src_language,target_languge)
                    if(line_translated == ""):
                        buff = buff + line
                    else:
                        buff = buff + line_translated + "\n"
                    # if(line[0] == '['):
                    #     buff = buff + '[' + lineTranslate + ']\n'
                    # elif(line[0] == '('):
                    #     buff = buff + '(' + lineTranslate + ')\n'
                    # elif(line[0] == '"'):
                    #     buff = buff + '"' + lineTranslate + '"\n'
                    # elif(lineTranslate == ""):
                    #     buff = buff + line
                    # else:
                    #     buff = buff + lineTranslate + "\n"
                except Exception as e:
                    print("[-] [{}] Failed to translate, keeping it".format(line))
                    buff = buff + line      #attach original line
                    continue
            else:
                buff = buff + line
            bar.next()
    src_file.close()

    if(target_languge == "ar"):
        dst_file_handle = open(dst_full,'w',encoding='utf-8')
    else:
        dst_file_handle = open(dst_full,'w')
    dst_file_handle.write(buff)
    dst_file_handle.close()

    return

def is_media_processed(full_file_no_ext):
    if(is_file_available(full_file_no_ext + "." + target_languge + "-auto.srt")):
        if(is_file_available(full_file_no_ext + "." + src_language + "-auto-" + model + ".srt")):
            if(is_file_available(full_file_no_ext + "." + target_languge + "-auto-" + model + ".srt")):
                return True
    return False


@app.route("/webhook", methods=["POST"])
def receive_webhook():
    event = ""
    if request.headers.get("source") == "Tautulli":
        event = request.json.get("event")

    if ((event == "library.new" or event == "added")) or ((event == "media.play" or event == "played")):
        print("[*] Webhook received!")
        full_file = request.json.get("file")
        full_file_no_ext = os.path.dirname(full_file) + "\\" + pathlib.Path(full_file).name.replace(pathlib.Path(full_file).suffix, "")
        full_path = os.path.dirname(full_file)

        if(is_media_processed(full_file_no_ext)):
            print("[*] Media already processed!.")
            return
        
        if(is_file_available(full_file_no_ext + ".audio.wav") == False):
            # Produce audio file
            prod_audio(full_file, full_file_no_ext + ".audio.wav")
        if (is_file_available(full_file_no_ext + "." + src_language + ".srt") and is_file_available(full_file_no_ext + "." + target_languge + "-auto-" + subtitle_custom_ext + ".srt") == False):
            # Translate existing subtitle
            translate(full_file_no_ext + "." + src_language + ".srt" , full_file_no_ext + "." + target_languge + "-auto-" + subtitle_custom_ext + ".srt")
        if(is_file_available(full_file_no_ext + "." + src_language + "-auto-" + model + ".srt") == False and is_file_available(full_file_no_ext + ".audio.wav")):
            # Create subtitle
            prod_subtitle(full_path, full_file_no_ext + "." + src_language + "-auto-" + model + ".srt" , full_file_no_ext + ".audio.wav" )
        if (is_file_available(full_file_no_ext + "." + src_language + "-auto-" + model + ".srt") and is_file_available(full_file_no_ext + "." + target_languge + "-auto-" + model + ".srt") == False):
            # Translate created subtitle
            translate(full_file_no_ext + "." + src_language + "-auto-" + model + ".srt" + "." + target_languge + "-auto-" + model + ".srt")

    return ""

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(webhookport))

