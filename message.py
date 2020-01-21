from datetime import datetime
import time

class Message:
    """
        Class to hold the message structure. 
            - content: str, text body
            - person: str, who sent the text in conversation
            - datetime: int, time the text is sent in seconds
            - datetime_str: str, time the text is sent in human readable way
    """
    def __init__(self):
        self.content = ''
        self.person = ''
        self.datetime = 0
        self.datetime_str = ''

def parse_datetime(time_string):
    """
    parse the time_string which is in the format:
    'yyyy-mm-dd, hour:minute:seconds AM/PM' into seconds
    """
    date = datetime.strptime(time_string, '%Y-%m-%d, %I:%M:%S %p')
    time_in_seconds = time.mktime(date.timetuple())
    return time_in_seconds

def read_whatsapp_chat_file(filename):
    """
        This is the Whatsapp conversation file obtained by
        clicking on a person's profile on conversation window 
        and -> Export Chat
        Though this parsing technique is implemented to extract
        texts from Whatsapp conversations. The techniques used here can 
        be extended to any kind of mesaging tools 
        which allow users to download their conversations 

        Here convert the chat file into list of instances of `Message`
        making sure that the order of conversation is preserved.
    """
    messages = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            # messages with attachments have extra weird character at the start
            # handle that case
            if 'attached' in line:
                line = line[line.find('['):]
            
            if line.startswith('['):
                # here we parse the message
                msg = Message() 
                tokens = line.split(': ', 1)
                if len(tokens) > 1: 
                    # it is a new message
                    msg.content = tokens[1].strip()
                    meta_tokens = tokens[0].split('] ', 1)
                    person = meta_tokens[1].strip()
                    msg.person = person
                    msg.datetime = parse_datetime(meta_tokens[0][1:].strip())
                    msg.datetime_str = meta_tokens[0][1:].strip()

                messages.append(msg)
            else:
                # it is a multipart message, combine this message with old one
                # assuming that the first message always has a timestamp with `[`
                messages[-1].content = '{}\n{}'.format(messages[-1].content, line)

    return messages
