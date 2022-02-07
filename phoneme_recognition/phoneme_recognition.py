import pandas as pd
import wave
import json

from vosk import Model, KaldiRecognizer, SetLogLevel

class Word:
    ''' A class representing a word from the JSON format for vosk speech recognition API '''

    def __init__(self, dict):
        '''
        Parameters:
          dict (dict) dictionary from JSON, containing:
            conf (float): degree of confidence, from 0 to 1
            end (float): end time of the pronouncing the word, in seconds
            start (float): start time of the pronouncing the word, in seconds
            word (str): recognized word
        '''

        self.conf = dict["conf"]
        self.end = dict["end"]
        self.start = dict["start"]
        self.word = dict["word"]

    def to_string(self):
        ''' Returns a string describing this instance '''
        return "{:20} from {:.2f} sec to {:.2f} sec, confidence is {:.2f}%".format(
            self.word, self.start, self.end, self.conf*100)

def wordToLetter(word):
    letter_len = (word.end-word.start)/len(word.word)
    letter_list = []
    i = 0
    while i < len(word.word):
        if word.word[i] == 'a' or word.word[i] == 'e' or word.word[i] == 'i':
            letter = ['a', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'o':
            letter = ['o', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'd' or word.word[i] == 'n' or word.word[i] == 'x' or word.word[i] == 'y' or word.word[i] == 'z':
            letter = ['c', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'g' or word.word[i] == 'k':
            letter = ['g', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'l':
            letter = ['l', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'b' or word.word[i] == 'm' or word.word[i] == 'p':
            letter = ['b', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'f' or word.word[i] == 'v':
            letter = ['f', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'j':
            letter = ['j', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'u':
            letter = ['u', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'q' or word.word[i] == 'w':
            letter = ['q', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
            letter_list.append(letter)
        elif word.word[i] == 'c':
            if i < (len(word.word)-1) and word.word[i+1] == 'h':
                letter = ['j', word.start+(i*letter_len), word.start+((i+2)*letter_len)]
                letter_list.append(letter)
                i += 1
            else :
                letter = ['c', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
                letter_list.append(letter)
        elif word.word[i] == 's':
            if i < (len(word.word)-1) and word.word[i+1] == 'h':
                letter = ['j', word.start+(i*letter_len), word.start+((i+2)*letter_len)]
                letter_list.append(letter)
                i += 1
            else :
                letter = ['c', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
                letter_list.append(letter)
        elif word.word[i] == 't':
            if i < (len(word.word)-1) and word.word[i+1] == 'h':
                letter = ['t', word.start+(i*letter_len), word.start+((i+2)*letter_len)]
                letter_list.append(letter)
                i += 1
            else :
                letter = ['c', word.start+(i*letter_len), word.start+((i+1)*letter_len)]
                letter_list.append(letter)
        i += 1
    return letter_list

def audioToLetters(model_path, audio_filename):
    model = Model(model_path)
    wf = wave.open(audio_filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    # get the list of JSON dictionaries
    results = []
    # recognize speech using vosk model
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            results.append(part_result)
    part_result = json.loads(rec.FinalResult())
    results.append(part_result)

    # convert list of JSON dictionaries to list of 'Word' objects
    list_of_Words = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition 
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for obj in sentence['result']:
            w = Word(obj)  # create custom Word object
            list_of_Words.append(w)  # and add it to list

    wf.close()  # close audiofile

    result_list = []
    for elt in list_of_Words:
        letter_list = wordToLetter(elt)
        for letter in letter_list:
            result_list.append(letter)
    return result_list

def phoneme_csv_creation(audio):
    model_path = "C:/Users/Enzo.Magal/Documents/Enzo2021/vosk-model-en-us-0.21"

    letter_list = audioToLetters(model_path, audio)
    index = list(range(len(letter_list)))
    df = pd.DataFrame(0, index=index,columns=['a', 'o', 'c', 'g', 'l', 'b', 'f', 't', 'j', 'u', 'q', 'start', 'end'])

    i = 0
    for elt in letter_list:
        df.loc[i, elt[0]] = 1
        df.loc[i, 'start'] = int(elt[1]*24)
        df.loc[i, 'end'] = int(elt[2]*24)
        i += 1
    index2 = list(range(df.loc[len(df)-1, 'end']))
    df_frame =  pd.DataFrame(0, index=index2, columns=['frame', 'a', 'o', 'c', 'g', 'l', 'b', 'f', 't', 'j', 'u', 'q'])
    for i in range(len(df_frame)):
        df_frame.loc[i, 'frame'] = '{:03}'.format(i+1)
    for i in range(len(df)):
        for y in range(df.loc[i, 'start'], df.loc[i, 'end']):
            df_frame.loc[y, 'a'] = df.loc[i, 'a']
            df_frame.loc[y, 'o'] = df.loc[i, 'o']
            df_frame.loc[y, 'c'] = df.loc[i, 'c']
            df_frame.loc[y, 'g'] = df.loc[i, 'g']
            df_frame.loc[y, 'l'] = df.loc[i, 'l']
            df_frame.loc[y, 'b'] = df.loc[i, 'b']
            df_frame.loc[y, 'f'] = df.loc[i, 'f']
            df_frame.loc[y, 't'] = df.loc[i, 't']
            df_frame.loc[y, 'j'] = df.loc[i, 'j']
            df_frame.loc[y, 'u'] = df.loc[i, 'u']
            df_frame.loc[y, 'q'] = df.loc[i, 'q']
    return df_frame