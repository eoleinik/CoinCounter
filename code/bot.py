import sys
import time
from pprint import pprint
import telepot
import another
from secret import TOKEN

bot = telepot.Bot(TOKEN)

def handle(msg):
    pprint(msg)
    content_type, chat_type, chat_id = telepot.glance(msg)
    try:
        if content_type in ['photo', 'document']:
            if content_type == 'photo':
                file_id = msg['photo'][-1]['file_id']
            else:
                file_id = msg['document']['file_id']
            file_name = 'received/temp.jpg'
            bot.downloadFile(file_id, file_name)
            num_coins, coin_sum = another.count_coins(file_name)
            bot.sendMessage(chat_id, 'Coin(s): {}\nEstimated sum: {} EUR'.format(num_coins, coin_sum))

            with open('received/processed.png', 'rb') as f:
                        bot.sendPhoto(chat_id, f)
        else:
            bot.sendMessage(chat_id, "Send me a photo of EURO coins, lying separately on a table, and I'll tell you how much there is.")
    #response = bot.sendPhoto(chat_id, minion_id)
    #pprint(response)
    except:
        print sys.exc_info()[0], sys.exc_info()[1]


bot.notifyOnMessage(handle)
print 'Listening ...'

# Keep the program running.
while 1:
    time.sleep(10)