
import asyncio

from time import time



async def print_nums():
    num=1
    while True:
        print(num)
        num+=1
        await asyncio.sleep(0.1)


async def print_time():
    count=0
    while True:
        if count %3 == 0:
            print("{} passed seconds", format(count))
        count += 1
        await asyncio.sleep(1)


async def main():
    task = asyncio.create_task(print_nums())
    task2 = asyncio.create_task(print_time())
    await asyncio.gather(*[task, task2])


def write_image(data):
    filename = f'file-{int(time() * 1000)}.jpg'
    with open(filename, 'wb') as file:
        file.write(data)

import aiohttp

async def fetch_content(url, session):
    async with session.get(url, allow_redirects=True) as response:
        data = await response.read()
        write_image(data)

async def main2():
    url = 'https://loremflickr.com/320/240'
    task = []
    async with aiohttp.ClientSession() as session:
        for i in range(10):
            task.append(asyncio.create_task(fetch_content(url, session)))
        await asyncio.gather(*task)

begin = time()
asyncio.run(main2())
end = time()
print(f'total  {end-begin}' )