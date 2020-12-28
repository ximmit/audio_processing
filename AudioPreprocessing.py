class Audioprocessor:
    def __init__(self):
        """Constructor"""


    def get_pitch_mono(self, audio_for_crepe, samplerate_for_crepe):

        self.time, self.frequency, self.confidence, self.activation = \
            crepe.predict(librosa.load(audio_for_crepe, sr=samplerate_for_crepe, mono=1), samplerate_for_crepe, viterbi=True)
        return self.time, self.frequency, self.confidence, self.activation

    def get_audio_data_mono_convert (self, audio_mono, samplerate_mono):

        self.audio = audio_mono
        self.data, self.samplerate = librosa.load(self.audio, sr=samplerate_mono, mono=1)
        self.data=np.asarray(self.data)
        return self.data

    def batch_audio_data_mono_processing(self, audio_mono, samplerate_mono, seconds_of_example):
        self.audio = audio_mono
        self.data, self.samplerate = librosa.load(self.audio, sr=samplerate_mono, mono=1)
        self.data=np.asarray(self.data)
        self.how_many_parts_in_audio=int(np.rint(len(self.data)/samplerate_mono * seconds_of_example))
        self.end_of_array=samplerate_mono * seconds_of_example * self.how_many_parts_in_audio
        self.len_of_example=samplerate_mono * seconds_of_example
        self.new_data=self.data[:self.end_of_array]
        self.batch_data=self.new_data.reshape(self.how_many_parts_in_audio, self.len_of_example)

        return self.batch_data
