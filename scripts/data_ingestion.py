import os
import telethon
from telethon import TelegramClient
import pandas as pd

# Telegram API credentials
api_id = '18471595'  # Replace with your API ID
api_hash = 'c28772a03a85f2ff424ef1bf0c26500c'  # Replace with your API Hash

# List of channels
channels = [
    '@Shageronlinestore',
    '@ZemenExpress',
    '@sinayelj',
    '@MerttEka',
    '@yebatochmregagroup',
    '@helloomarketethiopia',
    '@Leyueqa',
    '@kstoreaddis',
    '@Fashiontera',
]

# Initialize the Telegram client
client = TelegramClient('session_name', api_id, api_hash)

async def collect_messages(channel_name):
    async with client:
        messages = []
        async for message in client.iter_messages(channel_name, limit=100):  # Customize limit
            if message.message:  # Only text messages
                messages.append({'sender': message.sender_id, 'timestamp': message.date, 'channel': channel_name, 'text': message.message})
        
        # Save messages to a DataFrame
        df = pd.DataFrame(messages)
        
        # File path
        file_path = '../data/raw/telegram_messages.csv'
        
        # Save to CSV: append mode, avoid overwriting
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)  # Create file if it doesn't exist
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # Append to file without rewriting headers

        print(f"Messages from {channel_name} saved to {file_path}")

# Run the client to collect messages from each channel
client.start()

for channel in channels:
    client.loop.run_until_complete(collect_messages(channel))
