One can obtain the text conversations from WhatsApp
by clicking on a person's profile in conversation window 
and -> Export Chat, lets name it "_chat.txt".

Though only parser to extract texts from Whatsapp conversations
is implemented currently. The techniques used here can be extended to any kind 
of mesaging platforms which allow users to download their conversations 

First convert the chat file into list of instances of `Message`
making sure that the order of conversation is preserved.
```python
from message import read_whatsapp_chat_file
messages = read_whatsapp_chat_file('_chat.txt')
```

We can pass the list of `Message` objects to `Analyze` class
and calculate various stats that are pre-implemented. The methods implemented
for calculation of stats are agnostic to the messaging platform, because
the data structure they use is `Message`.

```python
from analyze import Analyze

chat = Analyze(messages)
print(chat.number_of_messages)
print('--------------------------')
print(chat.number_of_words)
print('--------------------------')
print(chat.number_of_questions)
print('--------------------------')
print(chat.n_frequent_words)
print('--------------------------')
print(chat.common_words)
print('--------------------------')
print(chat.one_word_replies)
print('--------------------------')
print(chat.average_resp_times)
print('--------------------------')
print(chat.max_delays)
print('--------------------------')
print(chat.nr_conv_starts)
print('--------------------------')
print(chat.nr_emojis)
print('--------------------------')
print(chat.emojis)
print('--------------------------')
print(chat.common_emojis)
print('--------------------------')
print(chat.nr_unique_emojis)
print('--------------------------')
print(chat.nr_messages_per_window)
print('--------------------------')
```
