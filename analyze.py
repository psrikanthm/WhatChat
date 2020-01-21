import re
from fuzzywuzzy import fuzz
import numpy as np
from collections import Counter, OrderedDict, defaultdict
import spacy
from nltk.corpus import words
from emoji import UNICODE_EMOJI

class OrderedCounter(Counter, OrderedDict):
    """
        An ordered dictionary can be combined with the Counter class
        so that the counter remembers the order elements are first
        encountered. Read more:
        https://cpython-test-docs.readthedocs.io/en/latest/library/collections.html#collections.OrderedDict
    """
    pass

class Analyze:
    """
        Supplies bunch of functions to analyze the text conversations
    """
    def __init__(self, messages):
        """
        Args:
            messages: [`message.Message`], conversation as the list of 
                            the objects of Message class after parsing
        """

        self.messages = messages
        
        # load the messages per person into dictionary
        self.messages_dict = {}
        for msg in messages:
            if len(msg.person) < 1:
                continue
            self.messages_dict[msg.person] = self.messages_dict.get(msg.person, 
                                            []) + [msg]
        
        # word limit for each text after which is it is considered a forward text
        self.word_limit = 40

        # filter out texts which contain image attachments and very long texts
        self.filter_messages_dict = {p: [msg.content for msg in messages if self.__filter_messages(msg.content)]
                                    for p,messages in self.messages_dict.items()}
        
        # split the filtered text sentences into words
        self.words_dict = {p: [k for item in [msg.split() for msg in msgs] for k in item]
                                for p,msgs in self.filter_messages_dict.items()}

        # setup stopwords
        nlp_en = spacy.load('en_core_web_lg')
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS

        self.zig_zag_messages = []
        prev_person = messages[0].person
        prev_time = messages[0].datetime
        for m in messages:
            cur_person = m.person
            cur_time = m.datetime
            if prev_person == cur_person:
                continue
            prev_person = cur_person
            self.zig_zag_messages.append(m)

        # u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        #self.emoji_pattern = re.compile(
        #    "(\ud83d[\ude00-\ude4f])|"  # emoticons
        #    "(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        #    "(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        #    "(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        #    "(\ud83c[\udde0-\uddff])"  # flags (iOS)
        #    "+")
        
        # load the vocabulary
        self.dictionary = set(words.words())

    def __filter_messages(self, textString):
        """
            Internal function: to filter out either very long texts (typically forwards)
            or images shared
        """
        if 'attached' in textString:
            return False

        if len(textString.split()) > self.word_limit:
            return False

        return True
    
    def __filter_words(self, words):
        filtered_words = []

        for wr in words:
            wr = wr.lower()
            
            # ignore email ids and links
            if '@' in wr:
                continue
            if 'http' in wr:
                continue
            
            # strip emojis attached to word without space
            m = re.match(r'\w+', wr)
            if m:
                w = m.group()
            else:
                continue
            
            if w in self.stopwords:
                continue
            if not w.isalpha():
                continue
            if len(w) <= 2:
                continue
            if w in self.dictionary:
                continue
            filtered_words.append(w)
        return filtered_words

    @staticmethod
    def sort_by_frequency(l):
        counter = OrderedCounter(l)

        freq_items = defaultdict(list)
        for item, freq in counter.items():
            freq_items[freq].append(item)

        ordered = []
        for f in sorted(freq_items, reverse=True):
            ordered += freq_items[f]

        return [(x, counter[x]) for x in ordered]

    @property
    def number_of_messages(self):
        """
            Count the number of messages from each person and
            return the messages per each as dictionary
        """
        nr_messages = {k: len(v) for k,v in self.messages_dict.items()}
        return nr_messages

    @property
    def number_of_words(self):
        """
            Count the number of words used by each person in the 
            conversation.
        """
        nr_words = {p: len(v) for p,v in self.words_dict.items()}
        return nr_words

    @property
    def number_of_questions(self):
        """
            number of questions from each
            sentences ending with '?' or sentences starting with
            'why', 'what', 'where', 'who', 'when', 'how'
        """
        def is_question(textString):
            sents = textString.split('\n')
            ques = False
            for sent in sents:
                sent = sent.strip()
                sent = sent.lower()
                if len(sent) == 0:
                    continue
                if any([sent.startswith(w) for w in [u'why', u'what', u'where', u'who', u'when', u'how']]):
                    ques = True
                if sent[-1] == '?':
                    ques = True
                if ques:
                    break
            return ques
        
        nr_questions = {p: len([m for m in msgs if is_question(m)])
                            for p,msgs in self.filter_messages_dict.items()}
        return nr_questions

    @property
    def n_frequent_words(self, n=20):
        """
            `n` frequent words used by each person
        """
        topwords_dict = {}
        for person, words in self.words_dict.items():
            # we need to remove the stopwords first
            filter_words = self.__filter_words(words)
            
            # now count the frequency of each word
            topwords_dict[person] = self.sort_by_frequency(filter_words)[:n]

        return topwords_dict

    @property
    def common_words(self, n=200):
        """
            Words that are most common between top `n` words of 
            all the speakers
        """
        words_lists = []
        for person, words in self.words_dict.items():
            # we need to remove the stopwords first
            filter_words = self.__filter_words(words)
            top_words = [w for w ,_ in self.sort_by_frequency(filter_words)[:n]]

            words_lists.append(set(top_words))
        return set.intersection(*words_lists)

    @property
    def one_word_replies(self):
        """
            Number of one word replies from each participant
        """
        def is_one_word(message):
            message = message.strip()
            if len(message.split(' ')) == 1:
                if message.isalpha():
                    return True
                else:
                    return False
            else:
                return False
        
        nr_words = {}
        for person, filter_messages in self.filter_messages_dict.items():
            nr_words[person] = len([m for m in filter_messages if is_one_word(m)])
        return nr_words

    @property
    def average_resp_times(self):
        """
            Average response time for each participant in the conversation
        """
        resp_times = {k:[] for k in self.filter_messages_dict.keys()}
        prev_time = self.zig_zag_messages[0].datetime

        for z in self.zig_zag_messages[1:]:
            cur_time = z.datetime
            delay = cur_time - prev_time
            prev_time = cur_time

            if delay > 3600 or delay < 0:
                continue

            resp_times[z.person].append(delay)

        avg_resp_times = {k:np.mean(v) if len(v) > 0 else 0 
                                for k,v in resp_times.items()}
        return avg_resp_times

    @property
    def max_delays(self, n=20):
        """
            Maximum delays upto `n` without texting each others in hours
            (the idle times in the conversation window)
        """
        prev_m = self.messages[0]
        prev_time = prev_m.datetime
        delays = []

        for m in self.messages[1:]:
            cur_time = m.datetime
            delay = cur_time - prev_time

            delays.append(delay)
            prev_time = cur_time

        delays = np.array(delays)
        arg_i = np.argpartition(delays, -n)[-n:]
        arg_i_sort = arg_i[np.argsort(delays[arg_i])][::-1]

        sorted_delays = [round(x/3600.0,2) for x in delays[arg_i_sort]]
        return sorted_delays

    @property
    def nr_conv_starts(self):
        """
            Number of conversation starters from each person
        """
        # definition of conversation starter: somebody who did first message after time window
        
        # here time window is defined to be 5 hours
        time_window = 5 * 3600

        conv_starts = {k:[] for k in self.filter_messages_dict.keys()}

        prev_time = self.messages[0].datetime
        for m in self.messages[1:]:
            cur_time = m.datetime
            delay = cur_time - prev_time
            prev_time = cur_time
            if delay >= time_window:
                conv_starts[m.person].append(m)
        
        return {k:len(v) for k,v in conv_starts.items()}

    @property
    def nr_emojis(self):
        """
            Number of emojis used by each participant
        """
        emojis_per_person = {k:[] for k in self.filter_messages_dict.keys()}
        for m in messages:
            emojis = []
            for emoji in UNICODE_EMOJI:
                emoji_count = m.content.count(emoji)
                emojis += ([emoji] * emoji_count)

            emojis_per_person[m.person] += emojis

        return {k: len(v) for k,v in emojis_per_person.items()}

    @property
    def emojis(self, n=20):
        """
            Top `n` emojis used by each person
        """
        emojis_per_person = {k:[] for k in self.filter_messages_dict.keys()}
        for m in messages:
            emojis = []
            for emoji in UNICODE_EMOJI:
                emoji_count = m.content.count(emoji)
                emojis += ([emoji] * emoji_count)
            emojis_per_person[m.person] += emojis

        top_emojis = {k: [w for w,_ in self.sort_by_frequency(v)[:n]] for
                                k,v in emojis_per_person.items()}
        return top_emojis

    @property
    def nr_unique_emojis(self):
        """
            Emojis that are most common to all participants
        """
        unique_emojis = {person:len(list(set(emojis))) for person, emojis in self.emojis.items()}
        return unique_emojis

    @property
    def common_emojis(self):
        """
            Emojis that are most common to all participants
        """
        emojis_lists = [set(emojis) for emojis in self.emojis.values()]
        return set.intersection(*emojis_lists)
        
if __name__ == '__main__':
    from message import read_whatsapp_chat_file
    messages = read_whatsapp_chat_file('_chat.txt')
    chat = Analyze(messages)
    #print(chat.number_of_messages)
    #print('--------------------------')
    #print(chat.number_of_words)
    #print('--------------------------')
    #print(chat.number_of_questions)
    #print('--------------------------')
    #print(chat.n_frequent_words)
    #print('--------------------------')
    #print(chat.common_words)
    #print('--------------------------')
    #print(chat.one_word_replies)
    #print('--------------------------')
    #print(chat.average_resp_times)
    #print('--------------------------')
    #print(chat.max_delays)
    #print('--------------------------')
    #print(chat.nr_conv_starts)
    #print('--------------------------')
    print(chat.nr_emojis)
    print('--------------------------')
    print(chat.emojis)
    print('--------------------------')
    print(chat.common_emojis)
    print('--------------------------')
    print(chat.nr_unique_emojis)
