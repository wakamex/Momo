# %% imports
# std lib
import asyncio
import importlib.util
import io
import json
import os
import random
import requests
import subprocess
import sys
from asyncio import Semaphore
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Union

# 3rd party
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from dotenv import dotenv_values
from pandas.plotting import table
from PIL import Image as PILImage
from PIL.Image import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# custom
import discord
from discord import CategoryChannel, DMChannel, Member, TextChannel, VoiceChannel
from discord.ext import commands, tasks
from discord.guild import Guild
from elf_sdk import calc_apr, calc_spot_price, get_apr, get_pool_config, get_pool_info, trade_to
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

MAX_MESSAGE_LENGTH = 2000

spec = importlib.util.spec_from_file_location("release_dates", "/code/scripts/release_dates.py")
release_dates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_dates)

GuildChannel = Union[TextChannel, VoiceChannel, CategoryChannel]

# pylint: disable=too-many-arguments
# ruff: noqa: D101, D102

# %% config
config = dotenv_values(".env")
assert config["DISCORD_BOT_TOKEN"], "DISCORD_BOT_TOKEN is not set in .env"


# %%
intents = discord.Intents.default()
intents.message_content = True
intents.members = True


class Momo(commands.Bot):
    guild: Guild
    channel: GuildChannel
    mihai_channel: TextChannel
    tcomp: TextChannel
    pool_info: dict
    pool_config: dict
    guild_members: list[Any]
    bg_task: dict[str, asyncio.Task] = {}
    rollbar_channels: dict[str, TextChannel] = {}
    rollbar_reported_ids: list[int] = []
    rollbar_reported_ids_testnet: list[int] = []

bot = Momo(command_prefix="$", intents=intents)

DELV_GUILD_ID = 754739461707006013
MIHAI_CHANNEL = 1028047253539147827
# ROBOTS_CHANNEL = 1035343815088816258  # rip old ro-bots
ROBOTS_CHANNEL = 1077979151316828270
DATASUSSY_CHANNEL = 1139538470696669215
HYPERDRIVE_CHANNEL = 1062553591988113489
MARCOMMS_CHANNEL = 798953513157132299
TCOMP_CHANNEL = 1133126205030273065
HYPERBOT_CHANNEL = 1230954647435612250
MIHAI_ID = 135898637422166016
CHART_UP = "<:darkmode_chart:1104081763170537533>"
CHART_DOWN = "<:darkmode_chart_down:1133555157406330930>"

delv_pfp: Image = PILImage.open("delvpfp.png").convert("RGB")

# %% commands


async def populate_recent_joiners(guild):
    user_data = {}
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    for member in tqdm(guild.members, desc="Populating recent joiners"):
        joined_at = member.joined_at
        if joined_at and joined_at > seven_days_ago:
            user_data[str(member.id)] = joined_at.isoformat()

    with open("members.json", "w", encoding="utf-8") as file:
        json.dump(user_data, file)


# async def is_imposter(member: Member, delv_pfp: Image, timeout=5):
#     avatar = str(member.display_avatar)

#     async with httpx.AsyncClient() as client:
#         response = await client.get(avatar, timeout=timeout)
#     avatar_img: Image = PILImage.open(io.BytesIO(response.content)).convert("RGB")

#     # Convert images to numpy arrays
#     avatar_img_np = np.array(avatar_img)
#     delv_pfp_np = np.array(delv_pfp)

#     # Ensure both images have the same dimensions
#     if avatar_img_np.shape != delv_pfp_np.shape:
#         return False

#     # Compute SSIM between two images
#     similarity_index = ssim(avatar_img_np, delv_pfp_np, multichannel=True)
#     print(f"{similarity_index=}")

#     # Return True if images are identical, False otherwise
#     return similarity_index == 1.0

async def find_member(guild: Guild, delv_pfp: Image, channel: GuildChannel):
    # member_display_name = "Element Finance® NOTICE#8822"
    # member = discord.utils.find(lambda m: m.display_name == member_display_name, guild.members)
    member_name_plus_discriminator = "mihai#3002"
    member = discord.utils.find(lambda m: f"{m.name}#{m.discriminator}" == member_name_plus_discriminator, guild.members)
    if member is None:
        print("Member not found")
    else:
        print(f"Found member {member.display_name} with ID {member.id} joined at {member.joined_at} role {member.top_role}")
        is_imposter_result = await is_imposter(member, delv_pfp)  # type: ignore
        print(f"They {'ARE' if is_imposter_result else 'ARE NOT'} an imposter")

async def report_imposter(member: Member, channel: GuildChannel):
    member_join_time = member.joined_at
    assert isinstance(member_join_time, datetime), "join time is not a datetime for {member.display_name}#{member.discriminator} ({member.id})"
    # print(f"Found member {member.display_name} with ID {member.id} joined at {member.joined_at} role {member.top_role}")
    # print(f"{member.display_name}#{member.discriminator} is an imposter! ID {member.id} joined {member_join_time.strftime('%d %B %Y')}")
    # Create an embed message
    embed = discord.Embed(title=f"{member.display_name} is an imposter!", description=f"Joined {member_join_time.strftime('%d %B %Y')}", color=discord.Color.red())
    embed.set_thumbnail(url=str(member.display_avatar))

    # Send the embed message to the channel
    await channel.send(embed=embed)


async def check_for_imposters(guild, channel: GuildChannel, atatime=5):
    semaphore: Semaphore = asyncio.Semaphore(atatime)  # Initialize a semaphore with a limit of 5 simultaneous tasks
    async with aiohttp.ClientSession() as session:
        tasks = [check_member(member, session, semaphore, channel) for member in tqdm(guild.members, desc="Checking for imposters") if member.display_avatar]
        for future in async_tqdm.as_completed(tasks, desc="Checking for imposters"):
            await future


async def get_member(member: Member, semaphore, session) -> dict:
    async with semaphore:
        return {
            "id": member.id,
            "name": member.name,
            "discriminator": member.discriminator,
            "joined_at": member.joined_at,
            "created_at": member.created_at,
            "avatar": member.avatar,
            "display_avatar": member.display_avatar,
            "display_name": member.display_name,
            "top_role": member.top_role,
        }


async def get_guild_members(atatime=5) -> pd.DataFrame:
    semaphore: asyncio.Semaphore = asyncio.Semaphore(atatime)  # Initialize a semaphore with a limit of 5 simultaneous tasks
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_member(member, semaphore, session)
            for member in bot.guild.members
            # if member.display_avatar
        ]
        guild_members = []
        for future in async_tqdm.as_completed(tasks, desc="Getting members"):
            result = await future
            guild_members.append(result)
        return pd.DataFrame(guild_members)


async def check_member(member: Member, session: ClientSession, semaphore, channel: GuildChannel):
    async with semaphore:
        is_member_imposter = await is_imposter(member, delv_pfp, session)
        if is_member_imposter:
            await report_imposter(member, channel)


async def is_imposter(member: Member, delv_pfp: Image, session=aiohttp.ClientSession(), timeout=5):
    avatar = str(member.display_avatar)
    try:
        async with session.get(avatar, timeout=timeout) as response:
            response_content = await response.read()
        avatar_img: Image = PILImage.open(io.BytesIO(response_content)).convert("RGB")
        return avatar_img == delv_pfp
    except asyncio.TimeoutError:
        print(f"Timeout error occurred while fetching avatar for {member.display_name}#{member.discriminator} ({member.id})")
        return False


@bot.event
async def on_member_join(member):
    joined_at = member.joined_at  # datetime object
    if joined_at > datetime.now(timezone.utc) - timedelta(days=7):  # Check if the member joined within the last 7 days
        user_data = {str(member.id): joined_at.isoformat()}  # Convert datetime to string
        if os.path.exists("members.json"):  # If the file already exists, load it and append the new member
            with open("members.json", "r", encoding="utf-8") as file:
                data = json.load(file)
            data.update(user_data)
            with open("members.json", "w", encoding="utf-8") as file:
                json.dump(data, file)
        else:  # If the file doesn't exist, create it and add the new member
            with open("members.json", "w", encoding="utf-8") as file:
                json.dump(user_data, file)

@bot.command(
    name="deps",
    description="Get dependencies",
    pass_context=True,
)
async def deps(context):
    if context.channel != bot.channel:
        await context.send("This command can only be used in the 🤖︱ro-bots channel.")
        return
    if context.guild != bot.guild:
        await context.send("This command can only be used in the DELV server.")
        return
    msg = await release_dates.get_release_dates("/home/mihai/.pyenv/versions/elf-env/bin/pip", short=True)
    await context.channel.send(msg)

@bot.command(
    name="imposters",
    description="Checks for imposters",
    pass_context=True,
)
async def imposters(context):
    if context.channel != bot.channel:
        await context.send("This command can only be used in the 🤖︱ro-bots channel.")
        return
    if context.guild != bot.guild:
        await context.send("This command can only be used in the DELV server.")
        return
    await check_for_imposters(context.guild, context.channel)


@bot.command(
    name="say",
    description="Tells you what to say",
    pass_context=True,
)
async def say(context, *args):
    print(f"context = {context}, context.channel = {context.channel}, context.channel.id = {context.channel.id}, args = {args}")
    if context.channel != bot.channel and context.channel.id != MARCOMMS_CHANNEL and context.channel.id != DATASUSSY_CHANNEL:
        print("not in the right channel")
        return
    if context.author.id != MIHAI_ID:
        response_list = ["You're not my mom! :angry:", "I only listen to Mihai! :dogegun:", "*bites a toe* :dogelick:", "You can't tell me what to do! :dogwhat:"]
        await context.send(random.choice(response_list))
        return
    await context.send(" ".join(args))

@bot.command(
    name="hyper",
    description="Tells you what to say in #hyperdrive",
    pass_context=True,
)
async def hyper(context, *args):
    print(f"context = {context}, context.channel = {context.channel}, context.channel.id = {context.channel.id}, args = {args}")
    hyperdrive_channel = bot.get_channel(HYPERDRIVE_CHANNEL)
    if context.author.id != MIHAI_ID:
        response_list = ["You're not my mom! :angry:", "I only listen to Mihai! :dogegun:", "*bites a toe* :dogelick:", "You can't tell me what to do! :dogwhat:"]
        await context.send(random.choice(response_list))
        return
    await hyperdrive_channel.send(" ".join(args))


@bot.command(
    name="hello",
    description="Sends a hello message",
    pass_context=True,
)
async def hello(context):
    await context.send("Hello!")


@bot.command(
    name="hardstyle",
    description="Plays a random hardstyle mp3 from Mihai's collection",
    pass_context=True,
)
async def hardstyle(context):
    user = context.author  # grab the user who sent the command
    voice_channel = user.voice.channel if user.voice else None
    channel = None
    # only play music if user is in a voice channel
    if voice_channel is not None:
        # grab user's voice channel
        channel = voice_channel.name
        # connect to voice channel and create AudioSource
        vc = await voice_channel.connect()
        # pick a random file from /data/mp3s
        file = random.choice(os.listdir("/data/mp3s"))
        await context.send(f"Playing {file[:-4]} in {channel}")
        source = discord.FFmpegPCMAudio(f"/data/mp3s/{file}")
        vc.play(source, after=lambda e: print("Player error: {e}") if e else None)
        while vc.is_playing():
            await asyncio.sleep(1)
        # disconnect after the player has finished
        vc.stop()
        await vc.disconnect()
    else:
        await context.send("User is not in a channel.")


async def save_guild_members_df():
    guild_members = await get_guild_members()
    guild_members_df = pd.DataFrame(guild_members)
    guild_members_df.to_csv("guild_members.csv", index=False)
    print("saved guild members to csv")

async def ban_member(id):
    try:
        await bot.guild.ban(discord.Object(id), reason="Spammers")
        print(f"Banned {id}")
    except Exception as exc:
        print(f"Failed to ban {id}: {exc}")

async def ban_ids_to_ban(atatime=5):
    ids_to_ban = pd.read_csv("ids_to_ban.csv")
    ids_to_ban = ids_to_ban["id"].values.tolist()
    semaphore = asyncio.Semaphore(atatime)
    async with semaphore:
        tasks = [ban_member(id) for id in ids_to_ban]
        for future in async_tqdm.as_completed(tasks, desc="Banning spammers"):
            await future

@bot.command(
    name="getPoolInfo",
    description="gets pool info",
    pass_context=True,
)
async def get_pool_info_discord(context):
    if context.channel != bot.channel and context.channel != bot.tcomp:
        await context.send("This command can only be used in the trading competition or 🤖︱ro-bots channel.")
        return
    pool_info = get_pool_info()
    await context.send(pool_info)

@bot.command(name="spotPrice",description="gets spot price",pass_context=True)
async def spot_price(context):
    if context.channel not in [bot.channel, bot.tcomp]:
        await context.send("This command can only be used in the trading competition or 🤖︱ro-bots channel.")
        return
    spot_price = get_spot(pool_config=bot.pool_config, pool_info=bot.pool_info)
    await context.send(f"Spot price: {spot_price}")

@bot.command(name="apr",description="gets apr",pass_context=True)
async def apr(context):
    if context.channel not in [bot.channel, bot.tcomp]:
        await context.send("This command can only be used in the trading competition or 🤖︱ro-bots channel.")
        return
    apr = get_apr(pool_config=bot.pool_config, pool_info=bot.pool_info)
    await context.send(f"APR: {apr}")

def get_spot(pool_config: dict, pool_info: dict) -> Decimal:
    return calc_spot_price(pool_config["initialSharePrice"], pool_info["shareReserves"], pool_info["shareAdjustment"], pool_info["bondReserves"], pool_config["timeStretch"])

def get_spot_diffs(pool_info) -> tuple[Decimal, Decimal, Decimal]:
    new_spot = get_spot(bot.pool_config, pool_info)
    old_spot = get_spot(bot.pool_config, bot.pool_info)
    diff_spot = new_spot - old_spot
    return new_spot, old_spot, diff_spot

def get_apr_diffs(pool_info) -> tuple[Decimal, Decimal, Decimal]:
    new_apr = get_apr(bot.pool_config, pool_info)
    old_apr = get_apr(bot.pool_config, bot.pool_info)
    diff_apr = new_apr - old_apr
    return new_apr, old_apr, diff_apr

def target_func(bond_reserves, target_apr, share_reserves, share_adjustment, initial_share_price, position_duration_days, time_stretch) -> Decimal:
    target_apr = Decimal(target_apr)
    apr = calc_apr(share_reserves, share_adjustment, bond_reserves, initial_share_price, position_duration_days, time_stretch)
    return apr - target_apr

@bot.command()
async def tradeto(context, target):
    if context.channel not in [bot.channel, bot.tcomp]:
        await context.send("This command can only be used in the trading competition or 🤖︱ro-bots channel.")
        return
    try:
        target = float(target)
    except TypeError:
        await context.send(f"Invalid target APR: {target}. Must be a number. For example 0.05 is a target APR of 5%.")
        return
    required_bonds = trade_to(target, pool_config=bot.pool_config, pool_info=bot.pool_info)
    await context.send(("SHORT" if required_bonds > 0 else "LONG") + f" {abs(required_bonds)} bonds for {target} APR")

async def update_reserves(bot: Momo):
    while not bot.is_closed():
        # Get the pool info
        pool_info = get_pool_info()

        # If the pool info have changed, send a message
        if {k: v for k, v in pool_info.items() if k != 'sharePrice'} != {k: v for k, v in bot.pool_info.items() if k != 'sharePrice'}:
            channel = bot.channel
            assert isinstance(channel, TextChannel), "Channel is not a TextChannel"
            # await channel.send(f'Pool info has changed: {pool_info}')
            new_spot, old_spot, diff_spot = get_spot_diffs(pool_info)
            new_apr, old_apr, diff_apr = get_apr_diffs(pool_info)
            spot_diff_str = (f"HIGHER{CHART_UP} " if diff_spot > 0 else f"LOWER{CHART_DOWN} " if diff_spot < 0 else "")
            spot_change_str = " " + (":arrow_up:" if diff_spot > 0 else "DOWN:arrow_down:" if diff_spot < 0 else "") + f" by {diff_spot} from {old_spot}" if diff_spot != 0 else ""
            apr_diff_str = (f"HIGHER{CHART_UP} " if diff_apr > 0 else f"LOWER{CHART_DOWN} " if diff_apr < 0 else "")
            apr_change_str = " " + (":arrow_up:" if diff_apr > 0 else "DOWN:arrow_down:" if diff_apr < 0 else "") + f" by {diff_apr} from {old_apr}" if diff_apr != 0 else ""
            # log_str = f"Spot price is now {spot_diff_str}at {new_spot}{spot_change_str}\nAPR is now {apr_diff_str}at {new_apr}{apr_change_str}"
            if abs(diff_apr - 0) < 1e-8:
                break
            log_str = f"Rate {('UP' if diff_apr > 0 else 'DOWN')} {diff_apr:.5%} to {new_apr:.5%}"
            await channel.send(log_str)
            bot.pool_info = pool_info
        else:
            print(f"{datetime.now()}: pool info unchanged. spot = {get_spot(bot.pool_config, bot.pool_info)}. apr = {get_apr(bot.pool_config, bot.pool_info)}.")

        await asyncio.sleep(1)  # run every second

# %%
@bot.command()
async def elf(context, target):
    if context.channel not in [bot.channel] and not isinstance(context.channel, DMChannel):
        await context.send("This command can only be used in the 🤖︱ro-bots channel.")
        return
    try:
        print(f"got elf command for elf #{target}, reading metadata.csv...", end="")
        from elfiverse import get_elf
        description_lines, elf = get_elf(target)
        embed2 = discord.Embed(title=f"Elf #{target} picture")
        print("done")
        print(f"getting elf #{target} image: {elf.image.iloc[0]}...", end="")
        embed2.set_image(url=elf.image.iloc[0])
        print("done")
        # await context.send(file=file, embeds=[embed1,embed2])
        await context.send('\n'.join(description_lines), embed=embed2)
    except Exception as exc:
        await context.send("Failed to get elf 🥴")
        await context.send(f"error: {exc}")
        return

@bot.command()
async def calc_costs(context, *args):
    if context.channel.id not in [HYPERBOT_CHANNEL] and not isinstance(context.channel, DMChannel):
        await context.send("This command can only be used in the 🤖 hyperbot thread.")
        return
    try:
        # result_message = await context.send("🤖 Generating response...")  # Initial placeholder message
        from vertex_function import calc_costs
        todays_date = datetime.now().date()
        calc_costs(
            project_id=config["PROJECT_ID"],
            start_time=todays_date,
            end_time=todays_date,
        )
    except Exception as exc:
        await context.send("Failed to calc_costs 🥴")
        await context.send(f"error: {exc}")
        return

@bot.command()
async def hyperdrive(context, *args):
    if context.channel.id not in [HYPERBOT_CHANNEL] and not isinstance(context.channel, DMChannel):
        await context.send("This command can only be used in the 🤖 hyperbot thread.")
        return
    try:
        # result_message = await context.send("🤖 Generating response...")  # Initial placeholder message
        from vertex_function import init_model, load_history, start_chat, send_message, save_history, respond_chunked
        model = init_model(project=config["PROJECT_ID"], location=config["LOCATION"])
        history = load_history()
        session = start_chat(model=model, history=history)
        # if len(history)==0:
        #     hyperdrive_codebase = open("/code/human3090/hyperdrive.txt", "r", encoding="utf-8").read()
        #     message = f"You are an expert AI assistant to help explain and improve the codebase of Hyperdrive, the next-gen interest rate AMM. Here is the full codebase:\n\n{hyperdrive_codebase}"
        #     print(f"=== SEEDING HISTORY ===\n{message}")
        #     response = send_message(session=session, message=message)
        #     for d in dir(response):
        #         if not d.startswith("_"):
        #             print(f" {d} = {getattr(response, d)}")
        #     full_response = response.text
        #     print(f"=== SEEDING_RESPONSE ===\n{full_response}")
        full_input = " ".join(args)
        print(f"=== INPUT ===\n{full_input}")
        # responses = send_message(session, full_input, stream=True)
        # full_response = "".join([response.text for response in responses])
        response = send_message(session=session, message=full_input)
        full_response = response.text
        print(f"=== RESPONSE ===\n{full_response}")
        save_history(session)

        # respond with an embed
        # embed = discord.Embed(
        #     title=full_input,
        #     description=full_response,
        #     color=discord.Color.blue()
            
        # )
        # await context.send(embed=embed)

        # respond with a message
        await respond_chunked(context, full_response)
        
    except Exception as exc:
        await context.send("Failed to Hyperdrive 🥴")
        await context.send(f"error: {exc}")
        return

@bot.command()
async def runs(context):
    if context.channel.id not in [ROBOTS_CHANNEL, DATASUSSY_CHANNEL]:
        await context.send("This command can only be used in the 🤪︱data-sussy or 🤖︱ro-bots.")
        return
    try:
        # count files named "results2*" in /code/experiments/experiments folder
        def count_files():
            folder_path = "/nvme/experiments/experiments"
            search_pattern = "results2*"
            find_command = f"find {folder_path} -type f -name {search_pattern}"

            completed_process = subprocess.run(find_command, shell=True, text=True, capture_output=True)
            if completed_process.returncode != 0:
                print("Error executing find command")
                return 0

            file_list = completed_process.stdout.split('\n')
            # Remove empty strings from the list
            file_list = [file for file in file_list if file]

            return len(file_list)

        # Inside the runs function
        num_runs = count_files()
        # count length of run_matrix.txt
        total_runs = len(open("/nvme/experiments/run_matrix.txt").readlines())
        await context.send(f"{num_runs}/{total_runs} runs")
    except Exception as exc:
        await context.send("Failed to count runs")
        return
    
async def background_task(observer):
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        observer.stop()
        observer.join()

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        # Place your logic here when a new file is detected
        asyncio.run_coroutine_threadsafe(bot.mihai_channel.send(f"New file detected: {event.src_path}"), bot.loop)
        # Send file to discord
        file_url_future = asyncio.run_coroutine_threadsafe(send_file_and_get_url(event.src_path), bot.loop)
        # get the file url
        file_url = file_url_future.result()  # This gets the URL from the future object
        # send image command to midjourney
        command = f"/imagine prompt:{file_url} rimworld pawn"
        asyncio.run_coroutine_threadsafe(bot.mihai_channel.send(command), bot.loop)

async def send_file_and_get_url(file_path):
    if bot.mihai_channel:
        with open(file_path, 'rb') as file:
            message = await bot.mihai_channel.send(file=discord.File(file, filename=os.path.basename(file_path)))
            return message.attachments[0].url if message.attachments else None

async def send_file_and_command(file_path, command):
    # Ensure that the bot's channel is set and valid
    if bot.channel:
        with open(file_path, 'rb') as file:
            # Send the file along with the command
            await bot.channel.send(command, file=discord.File(file, filename=os.path.basename(file_path)))

# %%
GET_GUILD_MEMBERS = False

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    assert (guild := bot.get_guild(DELV_GUILD_ID)), "Guild not found"
    bot.guild = guild
    assert (channel := bot.get_channel(ROBOTS_CHANNEL)), "Robots channel not found"
    assert isinstance(channel, TextChannel), "Channel is not a TextChannel"
    bot.channel = channel
    assert (channel := bot.get_channel(MIHAI_CHANNEL)), "Mihai channel not found"
    assert isinstance(channel, TextChannel), "Channel is not a TextChannel"
    bot.mihai_channel = channel
    
    if GET_GUILD_MEMBERS is True:
        df = await get_guild_members()
        df.to_csv("guild_members.csv", index=False)
        print("saved guild members to csv")
    
    # Pics
    # path = "/nvme/pics"  # Set the path to the directory you want to watch
    # event_handler = MyHandler()
    # observer = Observer()
    # observer.schedule(event_handler, path, recursive=False)
    # observer.start()
    # bot.bg_task["pics"] = bot.loop.create_task(background_task(observer))

    # trading comp
    # assert (tcomp := bot.get_channel(TCOMP_CHANNEL)), "Trading competition channel not found"
    # assert isinstance(tcomp, TextChannel), "Channel is not a TextChannel"
    # bot.tcomp = tcomp
    # bot.pool_config = get_pool_config()
    # bot.pool_info = get_pool_info()

    # await update_reserves(bot)

    # await ban_ids_to_ban()
    # await populate_recent_joiners(guild)
    # await find_member(guild, delv_pfp, channel)


# %% run it
bot.run(config["DISCORD_BOT_TOKEN"])
