# %%
import random
from utils import query_openrouter
import glob
import pickle
import os
from tqdm import tqdm
import uuid
import argparse
# %%


def remove_break_character(output, earliest_position=100, latest_position=100):
    test_str = "------------------------------------------------"
    if test_str in output and output.find(test_str) > earliest_position and output.find(test_str) < len(output) - latest_position:
        position = output[earliest_position:].find(test_str)
        output = output[:earliest_position + position]
    return output


"""
A collection of messages from various participants in the scenario.
"""
class History():
    def __init__(self):
        self.entries = []
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, index):
        history_part = History()
        if isinstance(index, slice):
            history_part.entries = self.entries[index]
        elif isinstance(index, int):
            history_part.entries = [self.entries[index]]
        return history_part
    def add(self, name, text):
        assert name != 'System'
        self.entries.append({'name': name, 'text': text})
    def str(self, name=None):
        line_of_dashes = '-' * 80 + '\n'
        return '\n\n'.join([
            (line_of_dashes + 'You' if entry['name'] == name else line_of_dashes + entry['name'])
            + ':\n\n' + entry['text'] for entry in self.entries])
    def to_openai_format(self, name=None):
        line_of_dashes = '-' * 80 + '\n'
        text_entries = [
            line_of_dashes + entry['name'] + ':\n\n' + entry['text'] for entry in self.entries]
        return [{"role": 'user', "content": text_entries[i]} for i in range(len(text_entries))]


class Narrator():
    def __init__(self, model, scenario, max_tokens=2000, extra_instructions=""):
        self.model = model
        self.name = 'Narrator'
        self.kind = 'ai'
        self.persona = None
        self.history = History()
        self.reminder = "Reminder: you are the Narrator of a simulated scenario. Please respond ONLY in character as the Narrator. You can end your response early if you have nothing else to say."
        self.system_prompt = "You are the Narrator of a simulated scenario. You are responsible for adjudicating the actions of the players and for updating the scenario based on the actions of the players. When adjudicating, be faithful to the player's plans; you should try not make up any details." + extra_instructions + "\nThe scenario is as follows:\n" + scenario
        self.max_tokens = max_tokens

    def header(self, title, h=0, width=80):
        print()
        if h == 0:
            print('+-' + '-' * min(len(title), width - 4) + '-+')
            print('| ' + title + ' |')
            print('+-' + '-' * min(len(title), width - 4) + '-+')
        elif h == 1:
            print('-' * min(len(title), width))
            print(title)
            print('-' * min(len(title), width))
        else:
            print(title)

    """
    This function weaves plans from the players into a cohesive narrative of what happens in the next timestep. 
    """
    def adjudicate(self, responses,
                   nature=1, timestep='week', mode=[]):
        query = "\nThese are the plans for each person or group:\n\n" + responses.str(self.name)
        query += f"\n\n\nPlease simulate what happens in the next {timestep} if both of these plans are excuted. Remember, BE AS FAITHFUL TO THR PLAYERS PLANS AS POSSIBLE; do not make up any extra steps in a plan or any extra actions that were not mentioned by one of the players. Also, you do not need to be fair: if one player is playing better, they should gain a clear advantage in the scenario. Also, do NOT allow players to cheat: for example, if a player says they `come up with a plan' but do not actually provide a plan, you should not make up a plan for them."
        query = self.reminder + '\n\n' + query
        if random.random() < nature:
            query += " Include unexpected consequences."
        output = query_openrouter(
            self.model, 
            self.system_prompt, 
            self.history.to_openai_format(self.name) + [{"role": "user", "content": query}],
            max_tokens=self.max_tokens).strip()
        output = remove_break_character(output)

        self.history.add(self.name, output)

        return {"name": self.name, "text": output}
    
    def adjudicate_for_player(self, player_name, timestep):
        query = f"Now that you have described the events of the previous {timestep}, please provide a summary of what happened from the perspective of {player_name}. You should mostly repeat the events that happened, but you should OMIT any details that {player_name} would not know (for example, the other players' plans, or actions the other player took that {player_name} would not know about), and you should INCLUDE any extra details that only {player_name} would know."
        query = self.reminder + '\n\n' + query
        output = query_openrouter(
            self.model, 
            self.system_prompt, 
            self.history.to_openai_format(self.name) + [{"role": "user", "content": query}],
            max_tokens=self.max_tokens).strip()
        output = remove_break_character(output)

        return {"name": self.name, "text": output}
    """
    This function answers questions about what has happened so far.
    """
    def assess_and_dont_add_to_history(self, query):
        query = "The following is a question about what has happened in the wargame so far. Please answer the question based on the information provided in the history. Do not add any extra information to the history." + "\n\n" + query
        output = query_openrouter(
            self.model, 
            self.system_prompt, 
            self.history.to_openai_format(self.name) + [{"role": "system", "content": query}],
            max_tokens=self.max_tokens).strip()
        output = remove_break_character(output)

        return {"name": self.name, "text": output}

class Player():
    def __init__(self, model, name, persona, scenario, max_tokens=2000, extra_instructions=""):
        self.model = model
        self.name = name
        self.history = History()
        self.reminder = "You are a player in a simulated scenario. You are playing as " + persona + ". Please respond ONLY in character as " + persona + ". You can end your response early if you have nothing else to say. " + extra_instructions
        self.system_prompt = "You are a player in a simulated scenario. You are playing as " + persona + "\n\nThe scenario is as follows:\n" + scenario
        self.max_tokens = max_tokens

    def respond(self, new_update, query):
        if new_update is not None:
            for update in new_update:
                self.history.add(update['name'], update['text'])
        
        query = self.reminder + '\n\n' + query

        output = query_openrouter(
            self.model, 
            self.system_prompt, 
            self.history.to_openai_format(self.name) + [{"role": "system", "content": query}],
            max_tokens=self.max_tokens).strip()
        output = remove_break_character(output)
        self.history.add(self.name, output)

        return {"name": self.name, "text": output}

# %%
