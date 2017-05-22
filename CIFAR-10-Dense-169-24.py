import tensorflow as tf
import os
from scipy import misc
import csv

initializer=tf.contrib.layers.xavier_initializer()

weights={
    'conv1': tf.get_variable('conv1', [7,7,3,24],tf.float32,initializer),

    'd1conv11': tf.get_variable('d1conv11w',[1,1,24,96],tf.float32,initializer),
    'd1conv12': tf.get_variable('d1conv12w',[3,3,96,24],tf.float32,initializer),
    'd1conv21': tf.get_variable('d1conv21w',[1,1,48,96],tf.float32,initializer),
    'd1conv22': tf.get_variable('d1conv22w',[3,3,96,24],tf.float32,initializer),
    'd1conv31': tf.get_variable('d1conv31w',[1,1,72,96],tf.float32,initializer),
    'd1conv32': tf.get_variable('d1conv32w',[3,3,96,24],tf.float32,initializer),
    'd1conv41': tf.get_variable('d1conv41w',[1,1,96,96],tf.float32,initializer),
    'd1conv42': tf.get_variable('d1conv42w',[3,3,96,24],tf.float32,initializer),
    'd1conv51': tf.get_variable('d1conv51w',[1,1,120,96],tf.float32,initializer),
    'd1conv52': tf.get_variable('d1conv52w',[3,3,96,24],tf.float32,initializer),
    'd1conv61': tf.get_variable('d1conv61w',[1,1,144,96],tf.float32,initializer),
    'd1conv62': tf.get_variable('d1conv62w',[3,3,96,24],tf.float32,initializer),

    't1conv': tf.get_variable('t1convw',[1,1,168,84],tf.float32,initializer),

    'd2conv11': tf.get_variable('d2conv11w',[1,1,84,96],tf.float32,initializer),
    'd2conv12': tf.get_variable('d2conv12w',[3,3,96,24],tf.float32,initializer),
    'd2conv21': tf.get_variable('d2conv21w',[1,1,108,96],tf.float32,initializer),
    'd2conv22': tf.get_variable('d2conv22w',[3,3,96,24],tf.float32,initializer),
    'd2conv31': tf.get_variable('d2conv31w',[1,1,132,96],tf.float32,initializer),
    'd2conv32': tf.get_variable('d2conv32w',[3,3,96,24],tf.float32,initializer),
    'd2conv41': tf.get_variable('d2conv41w',[1,1,156,96],tf.float32,initializer),
    'd2conv42': tf.get_variable('d2conv42w',[3,3,96,24],tf.float32,initializer),
    'd2conv51': tf.get_variable('d2conv51w',[1,1,180,96],tf.float32,initializer),
    'd2conv52': tf.get_variable('d2conv52w',[3,3,96,24],tf.float32,initializer),
    'd2conv61': tf.get_variable('d2conv61w',[1,1,204,96],tf.float32,initializer),
    'd2conv62': tf.get_variable('d2conv62w',[3,3,96,24],tf.float32,initializer),
    'd2conv71': tf.get_variable('d2conv71w',[1,1,228,96],tf.float32,initializer),
    'd2conv72': tf.get_variable('d2conv72w',[3,3,96,24],tf.float32,initializer),
    'd2conv81': tf.get_variable('d2conv81w',[1,1,252,96],tf.float32,initializer),
    'd2conv82': tf.get_variable('d2conv82w',[3,3,96,24],tf.float32,initializer),
    'd2conv91': tf.get_variable('d2conv91w',[1,1,276,96],tf.float32,initializer),
    'd2conv92': tf.get_variable('d2conv92w',[3,3,96,24],tf.float32,initializer),
    'd2conv101': tf.get_variable('d2conv101w',[1,1,300,96],tf.float32,initializer),
    'd2conv102': tf.get_variable('d2conv102w',[3,3,96,24],tf.float32,initializer),
    'd2conv111': tf.get_variable('d2conv111w',[1,1,324,96],tf.float32,initializer),
    'd2conv112': tf.get_variable('d2conv112w',[3,3,96,24],tf.float32,initializer),
    'd2conv121': tf.get_variable('d2conv121w',[1,1,348,96],tf.float32,initializer),
    'd2conv122': tf.get_variable('d2conv122w',[3,3,96,24],tf.float32,initializer),

    't2conv': tf.get_variable('t2convw',[1,1,372,186],tf.float32,initializer),

    'd3conv11': tf.get_variable('d3conv11w',[1,1,186,96],tf.float32,initializer),
    'd3conv12': tf.get_variable('d3conv12w',[3,3,96,24],tf.float32,initializer),
    'd3conv21': tf.get_variable('d3conv21w',[1,1,210,96],tf.float32,initializer),
    'd3conv22': tf.get_variable('d3conv22w',[3,3,96,24],tf.float32,initializer),
    'd3conv31': tf.get_variable('d3conv31w',[1,1,234,96],tf.float32,initializer),
    'd3conv32': tf.get_variable('d3conv32w',[3,3,96,24],tf.float32,initializer),
    'd3conv41': tf.get_variable('d3conv41w',[1,1,258,96],tf.float32,initializer),
    'd3conv42': tf.get_variable('d3conv42w',[3,3,96,24],tf.float32,initializer),
    'd3conv51': tf.get_variable('d3conv51w',[1,1,282,96],tf.float32,initializer),
    'd3conv52': tf.get_variable('d3conv52w',[3,3,96,24],tf.float32,initializer),
    'd3conv61': tf.get_variable('d3conv61w',[1,1,306,96],tf.float32,initializer),
    'd3conv62': tf.get_variable('d3conv62w',[3,3,96,24],tf.float32,initializer),
    'd3conv71': tf.get_variable('d3conv71w',[1,1,330,96],tf.float32,initializer),
    'd3conv72': tf.get_variable('d3conv72w',[3,3,96,24],tf.float32,initializer),
    'd3conv81': tf.get_variable('d3conv81w',[1,1,354,96],tf.float32,initializer),
    'd3conv82': tf.get_variable('d3conv82w',[3,3,96,24],tf.float32,initializer),
    'd3conv91': tf.get_variable('d3conv91w',[1,1,378,96],tf.float32,initializer),
    'd3conv92': tf.get_variable('d3conv92w',[3,3,96,24],tf.float32,initializer),
    'd3conv101': tf.get_variable('d3conv101w',[1,1,402,96],tf.float32,initializer),
    'd3conv102': tf.get_variable('d3conv102w',[3,3,96,24],tf.float32,initializer),
    'd3conv111': tf.get_variable('d3conv111w',[1,1,426,96],tf.float32,initializer),
    'd3conv112': tf.get_variable('d3conv112w',[3,3,96,24],tf.float32,initializer),
    'd3conv121': tf.get_variable('d3conv121w',[1,1,450,96],tf.float32,initializer),
    'd3conv122': tf.get_variable('d3conv122w',[3,3,96,24],tf.float32,initializer),
    'd3conv131': tf.get_variable('d3conv131w',[1,1,474,96],tf.float32,initializer),
    'd3conv132': tf.get_variable('d3conv132w',[3,3,96,24],tf.float32,initializer),
    'd3conv141': tf.get_variable('d3conv141w',[1,1,498,96],tf.float32,initializer),
    'd3conv142': tf.get_variable('d3conv142w',[3,3,96,24],tf.float32,initializer),
    'd3conv151': tf.get_variable('d3conv151w',[1,1,522,96],tf.float32,initializer),
    'd3conv152': tf.get_variable('d3conv152w',[3,3,96,24],tf.float32,initializer),
    'd3conv161': tf.get_variable('d3conv161w',[1,1,546,96],tf.float32,initializer),
    'd3conv162': tf.get_variable('d3conv162w',[3,3,96,24],tf.float32,initializer),
    'd3conv171': tf.get_variable('d3conv171w',[1,1,570,96],tf.float32,initializer),
    'd3conv172': tf.get_variable('d3conv172w',[3,3,96,24],tf.float32,initializer),
    'd3conv181': tf.get_variable('d3conv181w',[1,1,594,96],tf.float32,initializer),
    'd3conv182': tf.get_variable('d3conv182w',[3,3,96,24],tf.float32,initializer),
    'd3conv191': tf.get_variable('d3conv191w',[1,1,618,96],tf.float32,initializer),
    'd3conv192': tf.get_variable('d3conv192w',[3,3,96,24],tf.float32,initializer),
    'd3conv201': tf.get_variable('d3conv201w',[1,1,642,96],tf.float32,initializer),
    'd3conv202': tf.get_variable('d3conv202w',[3,3,96,24],tf.float32,initializer),
    'd3conv211': tf.get_variable('d3conv211w',[1,1,666,96],tf.float32,initializer),
    'd3conv212': tf.get_variable('d3conv212w',[3,3,96,24],tf.float32,initializer),
    'd3conv221': tf.get_variable('d3conv221w',[1,1,690,96],tf.float32,initializer),
    'd3conv222': tf.get_variable('d3conv222w',[3,3,96,24],tf.float32,initializer),
    'd3conv231': tf.get_variable('d3conv231w',[1,1,714,96],tf.float32,initializer),
    'd3conv232': tf.get_variable('d3conv232w',[3,3,96,24],tf.float32,initializer),
    'd3conv241': tf.get_variable('d3conv241w',[1,1,738,96],tf.float32,initializer),
    'd3conv242': tf.get_variable('d3conv242w',[3,3,96,24],tf.float32,initializer),
    'd3conv251': tf.get_variable('d3conv251w',[1,1,762,96],tf.float32,initializer),
    'd3conv252': tf.get_variable('d3conv252w',[3,3,96,24],tf.float32,initializer),
    'd3conv261': tf.get_variable('d3conv261w',[1,1,786,96],tf.float32,initializer),
    'd3conv262': tf.get_variable('d3conv262w',[3,3,96,24],tf.float32,initializer),
    'd3conv271': tf.get_variable('d3conv271w',[1,1,810,96],tf.float32,initializer),
    'd3conv272': tf.get_variable('d3conv272w',[3,3,96,24],tf.float32,initializer),
    'd3conv281': tf.get_variable('d3conv281w',[1,1,834,96],tf.float32,initializer),
    'd3conv282': tf.get_variable('d3conv282w',[3,3,96,24],tf.float32,initializer),
    'd3conv291': tf.get_variable('d3conv291w',[1,1,858,96],tf.float32,initializer),
    'd3conv292': tf.get_variable('d3conv292w',[3,3,96,24],tf.float32,initializer),
    'd3conv301': tf.get_variable('d3conv301w',[1,1,882,96],tf.float32,initializer),
    'd3conv302': tf.get_variable('d3conv302w',[3,3,96,24],tf.float32,initializer),
    'd3conv311': tf.get_variable('d3conv311w',[1,1,906,96],tf.float32,initializer),
    'd3conv312': tf.get_variable('d3conv312w',[3,3,96,24],tf.float32,initializer),
    'd3conv321': tf.get_variable('d3conv321w',[1,1,930,96],tf.float32,initializer),
    'd3conv322': tf.get_variable('d3conv322w',[3,3,96,24],tf.float32,initializer),

    't3conv': tf.get_variable('t3convw',[1,1,954,477],tf.float32,initializer),

    'd4conv11': tf.get_variable('d4conv11w',[1,1,477,96],tf.float32,initializer),
    'd4conv12': tf.get_variable('d4conv12w',[3,3,96,24],tf.float32,initializer),
    'd4conv21': tf.get_variable('d4conv21w',[1,1,501,96],tf.float32,initializer),
    'd4conv22': tf.get_variable('d4conv22w',[3,3,96,24],tf.float32,initializer),
    'd4conv31': tf.get_variable('d4conv31w',[1,1,525,96],tf.float32,initializer),
    'd4conv32': tf.get_variable('d4conv32w',[3,3,96,24],tf.float32,initializer),
    'd4conv41': tf.get_variable('d4conv41w',[1,1,549,96],tf.float32,initializer),
    'd4conv42': tf.get_variable('d4conv42w',[3,3,96,24],tf.float32,initializer),
    'd4conv51': tf.get_variable('d4conv51w',[1,1,573,96],tf.float32,initializer),
    'd4conv52': tf.get_variable('d4conv52w',[3,3,96,24],tf.float32,initializer),
    'd4conv61': tf.get_variable('d4conv61w',[1,1,597,96],tf.float32,initializer),
    'd4conv62': tf.get_variable('d4conv62w',[3,3,96,24],tf.float32,initializer),
    'd4conv71': tf.get_variable('d4conv71w',[1,1,621,96],tf.float32,initializer),
    'd4conv72': tf.get_variable('d4conv72w',[3,3,96,24],tf.float32,initializer),
    'd4conv81': tf.get_variable('d4conv81w',[1,1,645,96],tf.float32,initializer),
    'd4conv82': tf.get_variable('d4conv82w',[3,3,96,24],tf.float32,initializer),
    'd4conv91': tf.get_variable('d4conv91w',[1,1,669,96],tf.float32,initializer),
    'd4conv92': tf.get_variable('d4conv92w',[3,3,96,24],tf.float32,initializer),
    'd4conv101': tf.get_variable('d4conv101w',[1,1,693,96],tf.float32,initializer),
    'd4conv102': tf.get_variable('d4conv102w',[3,3,96,24],tf.float32,initializer),
    'd4conv111': tf.get_variable('d4conv111w',[1,1,717,96],tf.float32,initializer),
    'd4conv112': tf.get_variable('d4conv112w',[3,3,96,24],tf.float32,initializer),
    'd4conv121': tf.get_variable('d4conv121w',[1,1,741,96],tf.float32,initializer),
    'd4conv122': tf.get_variable('d4conv122w',[3,3,96,24],tf.float32,initializer),
    'd4conv131': tf.get_variable('d4conv131w',[1,1,765,96],tf.float32,initializer),
    'd4conv132': tf.get_variable('d4conv132w',[3,3,96,24],tf.float32,initializer),
    'd4conv141': tf.get_variable('d4conv141w',[1,1,789,96],tf.float32,initializer),
    'd4conv142': tf.get_variable('d4conv142w',[3,3,96,24],tf.float32,initializer),
    'd4conv151': tf.get_variable('d4conv151w',[1,1,813,96],tf.float32,initializer),
    'd4conv152': tf.get_variable('d4conv152w',[3,3,96,24],tf.float32,initializer),
    'd4conv161': tf.get_variable('d4conv161w',[1,1,837,96],tf.float32,initializer),
    'd4conv162': tf.get_variable('d4conv162w',[3,3,96,24],tf.float32,initializer),
    'd4conv171': tf.get_variable('d4conv171w',[1,1,861,96],tf.float32,initializer),
    'd4conv172': tf.get_variable('d4conv172w',[3,3,96,24],tf.float32,initializer),
    'd4conv181': tf.get_variable('d4conv181w',[1,1,885,96],tf.float32,initializer),
    'd4conv182': tf.get_variable('d4conv182w',[3,3,96,24],tf.float32,initializer),
    'd4conv191': tf.get_variable('d4conv191w',[1,1,909,96],tf.float32,initializer),
    'd4conv192': tf.get_variable('d4conv192w',[3,3,96,24],tf.float32,initializer),
    'd4conv201': tf.get_variable('d4conv201w',[1,1,933,96],tf.float32,initializer),
    'd4conv202': tf.get_variable('d4conv202w',[3,3,96,24],tf.float32,initializer),
    'd4conv211': tf.get_variable('d4conv211w',[1,1,957,96],tf.float32,initializer),
    'd4conv212': tf.get_variable('d4conv212w',[3,3,96,24],tf.float32,initializer),
    'd4conv221': tf.get_variable('d4conv221w',[1,1,981,96],tf.float32,initializer),
    'd4conv222': tf.get_variable('d4conv222w',[3,3,96,24],tf.float32,initializer),
    'd4conv231': tf.get_variable('d4conv231w',[1,1,1005,96],tf.float32,initializer),
    'd4conv232': tf.get_variable('d4conv232w',[3,3,96,24],tf.float32,initializer),
    'd4conv241': tf.get_variable('d4conv241w',[1,1,1029,96],tf.float32,initializer),
    'd4conv242': tf.get_variable('d4conv242w',[3,3,96,24],tf.float32,initializer),
    'd4conv251': tf.get_variable('d4conv251w',[1,1,1053,96],tf.float32,initializer),
    'd4conv252': tf.get_variable('d4conv252w',[3,3,96,24],tf.float32,initializer),
    'd4conv261': tf.get_variable('d4conv261w',[1,1,1077,96],tf.float32,initializer),
    'd4conv262': tf.get_variable('d4conv262w',[3,3,96,24],tf.float32,initializer),
    'd4conv271': tf.get_variable('d4conv271w',[1,1,1101,96],tf.float32,initializer),
    'd4conv272': tf.get_variable('d4conv272w',[3,3,96,24],tf.float32,initializer),
    'd4conv281': tf.get_variable('d4conv281w',[1,1,1125,96],tf.float32,initializer),
    'd4conv282': tf.get_variable('d4conv282w',[3,3,96,24],tf.float32,initializer),
    'd4conv291': tf.get_variable('d4conv291w',[1,1,1149,96],tf.float32,initializer),
    'd4conv292': tf.get_variable('d4conv292w',[3,3,96,24],tf.float32,initializer),
    'd4conv301': tf.get_variable('d4conv301w',[1,1,1173,96],tf.float32,initializer),
    'd4conv302': tf.get_variable('d4conv302w',[3,3,96,24],tf.float32,initializer),
    'd4conv311': tf.get_variable('d4conv311w',[1,1,1197,96],tf.float32,initializer),
    'd4conv312': tf.get_variable('d4conv312w',[3,3,96,24],tf.float32,initializer),
    'd4conv321': tf.get_variable('d4conv321w',[1,1,1221,96],tf.float32,initializer),
    'd4conv322': tf.get_variable('d4conv322w',[3,3,96,24],tf.float32,initializer),

    'fc': tf.get_variable('fcw',[1245,10],tf.float32,initializer),
}

biases={
    'conv1': tf.get_variable('conv1b',[24],tf.float32,initializer),

    'd1conv11': tf.get_variable('d1conv11b',[96],tf.float32,initializer),
    'd1conv12': tf.get_variable('d1conv12b',[24],tf.float32,initializer),
    'd1conv21': tf.get_variable('d1conv21b',[96],tf.float32,initializer),
    'd1conv22': tf.get_variable('d1conv22b',[24],tf.float32,initializer),
    'd1conv31': tf.get_variable('d1conv31b',[96],tf.float32,initializer),
    'd1conv32': tf.get_variable('d1conv32b',[24],tf.float32,initializer),
    'd1conv41': tf.get_variable('d1conv41b',[96],tf.float32,initializer),
    'd1conv42': tf.get_variable('d1conv42b',[24],tf.float32,initializer),
    'd1conv51': tf.get_variable('d1conv51b',[96],tf.float32,initializer),
    'd1conv52': tf.get_variable('d1conv52b',[24],tf.float32,initializer),
    'd1conv61': tf.get_variable('d1conv61b',[96],tf.float32,initializer),
    'd1conv62': tf.get_variable('d1conv62b',[24],tf.float32,initializer),

    't1conv': tf.get_variable('t1convb',[84],tf.float32,initializer),

    'd2conv11': tf.get_variable('d2conv11b',[96],tf.float32,initializer),
    'd2conv12': tf.get_variable('d2conv12b',[24],tf.float32,initializer),
    'd2conv21': tf.get_variable('d2conv21b',[96],tf.float32,initializer),
    'd2conv22': tf.get_variable('d2conv22b',[24],tf.float32,initializer),
    'd2conv31': tf.get_variable('d2conv31b',[96],tf.float32,initializer),
    'd2conv32': tf.get_variable('d2conv32b',[24],tf.float32,initializer),
    'd2conv41': tf.get_variable('d2conv41b',[96],tf.float32,initializer),
    'd2conv42': tf.get_variable('d2conv42b',[24],tf.float32,initializer),
    'd2conv51': tf.get_variable('d2conv51b',[96],tf.float32,initializer),
    'd2conv52': tf.get_variable('d2conv52b',[24],tf.float32,initializer),
    'd2conv61': tf.get_variable('d2conv61b',[96],tf.float32,initializer),
    'd2conv62': tf.get_variable('d2conv62b',[24],tf.float32,initializer),
    'd2conv71': tf.get_variable('d2conv71b',[96],tf.float32,initializer),
    'd2conv72': tf.get_variable('d2conv72b',[24],tf.float32,initializer),
    'd2conv81': tf.get_variable('d2conv81b',[96],tf.float32,initializer),
    'd2conv82': tf.get_variable('d2conv82b',[24],tf.float32,initializer),
    'd2conv91': tf.get_variable('d2conv91b',[96],tf.float32,initializer),
    'd2conv92': tf.get_variable('d2conv92b',[24],tf.float32,initializer),
    'd2conv101': tf.get_variable('d2conv101b',[96],tf.float32,initializer),
    'd2conv102': tf.get_variable('d2conv102b',[24],tf.float32,initializer),
    'd2conv111': tf.get_variable('d2conv111b',[96],tf.float32,initializer),
    'd2conv112': tf.get_variable('d2conv112b',[24],tf.float32,initializer),
    'd2conv121': tf.get_variable('d2conv121b',[96],tf.float32,initializer),
    'd2conv122': tf.get_variable('d2conv122b',[24],tf.float32,initializer),

    't2conv': tf.get_variable('t2conv',[186],tf.float32,initializer),

    'd3conv11': tf.get_variable('d3conv11b',[96],tf.float32,initializer),
    'd3conv12': tf.get_variable('d3conv12b',[24],tf.float32,initializer),
    'd3conv21': tf.get_variable('d3conv21b',[96],tf.float32,initializer),
    'd3conv22': tf.get_variable('d3conv22b',[24],tf.float32,initializer),
    'd3conv31': tf.get_variable('d3conv31b',[96],tf.float32,initializer),
    'd3conv32': tf.get_variable('d3conv32b',[24],tf.float32,initializer),
    'd3conv41': tf.get_variable('d3conv41b',[96],tf.float32,initializer),
    'd3conv42': tf.get_variable('d3conv42b',[24],tf.float32,initializer),
    'd3conv51': tf.get_variable('d3conv51b',[96],tf.float32,initializer),
    'd3conv52': tf.get_variable('d3conv52b',[24],tf.float32,initializer),
    'd3conv61': tf.get_variable('d3conv61b',[96],tf.float32,initializer),
    'd3conv62': tf.get_variable('d3conv62b',[24],tf.float32,initializer),
    'd3conv71': tf.get_variable('d3conv71b',[96],tf.float32,initializer),
    'd3conv72': tf.get_variable('d3conv72b',[24],tf.float32,initializer),
    'd3conv81': tf.get_variable('d3conv81b',[96],tf.float32,initializer),
    'd3conv82': tf.get_variable('d3conv82b',[24],tf.float32,initializer),
    'd3conv91': tf.get_variable('d3conv91b',[96],tf.float32,initializer),
    'd3conv92': tf.get_variable('d3conv92b',[24],tf.float32,initializer),
    'd3conv101': tf.get_variable('d3conv101b',[96],tf.float32,initializer),
    'd3conv102': tf.get_variable('d3conv102b',[24],tf.float32,initializer),
    'd3conv111': tf.get_variable('d3conv111b',[96],tf.float32,initializer),
    'd3conv112': tf.get_variable('d3conv112b',[24],tf.float32,initializer),
    'd3conv121': tf.get_variable('d3conv121b',[96],tf.float32,initializer),
    'd3conv122': tf.get_variable('d3conv122b',[24],tf.float32,initializer),
    'd3conv131': tf.get_variable('d3conv131b',[96],tf.float32,initializer),
    'd3conv132': tf.get_variable('d3conv132b',[24],tf.float32,initializer),
    'd3conv141': tf.get_variable('d3conv141b',[96],tf.float32,initializer),
    'd3conv142': tf.get_variable('d3conv142b',[24],tf.float32,initializer),
    'd3conv151': tf.get_variable('d3conv151b',[96],tf.float32,initializer),
    'd3conv152': tf.get_variable('d3conv152b',[24],tf.float32,initializer),
    'd3conv161': tf.get_variable('d3conv161b',[96],tf.float32,initializer),
    'd3conv162': tf.get_variable('d3conv162b',[24],tf.float32,initializer),
    'd3conv171': tf.get_variable('d3conv171b',[96],tf.float32,initializer),
    'd3conv172': tf.get_variable('d3conv172b',[24],tf.float32,initializer),
    'd3conv181': tf.get_variable('d3conv181b',[96],tf.float32,initializer),
    'd3conv182': tf.get_variable('d3conv182b',[24],tf.float32,initializer),
    'd3conv191': tf.get_variable('d3conv191b',[96],tf.float32,initializer),
    'd3conv192': tf.get_variable('d3conv192b',[24],tf.float32,initializer),
    'd3conv201': tf.get_variable('d3conv201b',[96],tf.float32,initializer),
    'd3conv202': tf.get_variable('d3conv202b',[24],tf.float32,initializer),
    'd3conv211': tf.get_variable('d3conv211b',[96],tf.float32,initializer),
    'd3conv212': tf.get_variable('d3conv212b',[24],tf.float32,initializer),
    'd3conv221': tf.get_variable('d3conv221b',[96],tf.float32,initializer),
    'd3conv222': tf.get_variable('d3conv222b',[24],tf.float32,initializer),
    'd3conv231': tf.get_variable('d3conv231b',[96],tf.float32,initializer),
    'd3conv232': tf.get_variable('d3conv232b',[24],tf.float32,initializer),
    'd3conv241': tf.get_variable('d3conv241b',[96],tf.float32,initializer),
    'd3conv242': tf.get_variable('d3conv242b',[24],tf.float32,initializer),
    'd3conv251': tf.get_variable('d3conv251b',[96],tf.float32,initializer),
    'd3conv252': tf.get_variable('d3conv252b',[24],tf.float32,initializer),
    'd3conv261': tf.get_variable('d3conv261b',[96],tf.float32,initializer),
    'd3conv262': tf.get_variable('d3conv262b',[24],tf.float32,initializer),
    'd3conv271': tf.get_variable('d3conv271b',[96],tf.float32,initializer),
    'd3conv272': tf.get_variable('d3conv272b',[24],tf.float32,initializer),
    'd3conv281': tf.get_variable('d3conv281b',[96],tf.float32,initializer),
    'd3conv282': tf.get_variable('d3conv282b',[24],tf.float32,initializer),
    'd3conv291': tf.get_variable('d3conv291b',[96],tf.float32,initializer),
    'd3conv292': tf.get_variable('d3conv292b',[24],tf.float32,initializer),
    'd3conv301': tf.get_variable('d3conv301b',[96],tf.float32,initializer),
    'd3conv302': tf.get_variable('d3conv302b',[24],tf.float32,initializer),
    'd3conv311': tf.get_variable('d3conv311b',[96],tf.float32,initializer),
    'd3conv312': tf.get_variable('d3conv312b',[24],tf.float32,initializer),
    'd3conv321': tf.get_variable('d3conv321b',[96],tf.float32,initializer),
    'd3conv322': tf.get_variable('d3conv322b',[24],tf.float32,initializer),

    't3conv': tf.get_variable('t3conv',[477],tf.float32,initializer),

    'd4conv11': tf.get_variable('d4conv11b',[96],tf.float32,initializer),
    'd4conv12': tf.get_variable('d4conv12b',[24],tf.float32,initializer),
    'd4conv21': tf.get_variable('d4conv21b',[96],tf.float32,initializer),
    'd4conv22': tf.get_variable('d4conv22b',[24],tf.float32,initializer),
    'd4conv31': tf.get_variable('d4conv31b',[96],tf.float32,initializer),
    'd4conv32': tf.get_variable('d4conv32b',[24],tf.float32,initializer),
    'd4conv41': tf.get_variable('d4conv41b',[96],tf.float32,initializer),
    'd4conv42': tf.get_variable('d4conv42b',[24],tf.float32,initializer),
    'd4conv51': tf.get_variable('d4conv51b',[96],tf.float32,initializer),
    'd4conv52': tf.get_variable('d4conv52b',[24],tf.float32,initializer),
    'd4conv61': tf.get_variable('d4conv61b',[96],tf.float32,initializer),
    'd4conv62': tf.get_variable('d4conv62b',[24],tf.float32,initializer),
    'd4conv71': tf.get_variable('d4conv71b',[96],tf.float32,initializer),
    'd4conv72': tf.get_variable('d4conv72b',[24],tf.float32,initializer),
    'd4conv81': tf.get_variable('d4conv81b',[96],tf.float32,initializer),
    'd4conv82': tf.get_variable('d4conv82b',[24],tf.float32,initializer),
    'd4conv91': tf.get_variable('d4conv91b',[96],tf.float32,initializer),
    'd4conv92': tf.get_variable('d4conv92b',[24],tf.float32,initializer),
    'd4conv101': tf.get_variable('d4conv101b',[96],tf.float32,initializer),
    'd4conv102': tf.get_variable('d4conv102b',[24],tf.float32,initializer),
    'd4conv111': tf.get_variable('d4conv111b',[96],tf.float32,initializer),
    'd4conv112': tf.get_variable('d4conv112b',[24],tf.float32,initializer),
    'd4conv121': tf.get_variable('d4conv121b',[96],tf.float32,initializer),
    'd4conv122': tf.get_variable('d4conv122b',[24],tf.float32,initializer),
    'd4conv131': tf.get_variable('d4conv131b',[96],tf.float32,initializer),
    'd4conv132': tf.get_variable('d4conv132b',[24],tf.float32,initializer),
    'd4conv141': tf.get_variable('d4conv141b',[96],tf.float32,initializer),
    'd4conv142': tf.get_variable('d4conv142b',[24],tf.float32,initializer),
    'd4conv151': tf.get_variable('d4conv151b',[96],tf.float32,initializer),
    'd4conv152': tf.get_variable('d4conv152b',[24],tf.float32,initializer),
    'd4conv161': tf.get_variable('d4conv161b',[96],tf.float32,initializer),
    'd4conv162': tf.get_variable('d4conv162b',[24],tf.float32,initializer),
    'd4conv171': tf.get_variable('d4conv171b',[96],tf.float32,initializer),
    'd4conv172': tf.get_variable('d4conv172b',[24],tf.float32,initializer),
    'd4conv181': tf.get_variable('d4conv181b',[96],tf.float32,initializer),
    'd4conv182': tf.get_variable('d4conv182b',[24],tf.float32,initializer),
    'd4conv191': tf.get_variable('d4conv191b',[96],tf.float32,initializer),
    'd4conv192': tf.get_variable('d4conv192b',[24],tf.float32,initializer),
    'd4conv201': tf.get_variable('d4conv201b',[96],tf.float32,initializer),
    'd4conv202': tf.get_variable('d4conv202b',[24],tf.float32,initializer),
    'd4conv211': tf.get_variable('d4conv211b',[96],tf.float32,initializer),
    'd4conv212': tf.get_variable('d4conv212b',[24],tf.float32,initializer),
    'd4conv221': tf.get_variable('d4conv221b',[96],tf.float32,initializer),
    'd4conv222': tf.get_variable('d4conv222b',[24],tf.float32,initializer),
    'd4conv231': tf.get_variable('d4conv231b',[96],tf.float32,initializer),
    'd4conv232': tf.get_variable('d4conv232b',[24],tf.float32,initializer),
    'd4conv241': tf.get_variable('d4conv241b',[96],tf.float32,initializer),
    'd4conv242': tf.get_variable('d4conv242b',[24],tf.float32,initializer),
    'd4conv251': tf.get_variable('d4conv251b',[96],tf.float32,initializer),
    'd4conv252': tf.get_variable('d4conv252b',[24],tf.float32,initializer),
    'd4conv261': tf.get_variable('d4conv261b',[96],tf.float32,initializer),
    'd4conv262': tf.get_variable('d4conv262b',[24],tf.float32,initializer),
    'd4conv271': tf.get_variable('d4conv271b',[96],tf.float32,initializer),
    'd4conv272': tf.get_variable('d4conv272b',[24],tf.float32,initializer),
    'd4conv281': tf.get_variable('d4conv281b',[96],tf.float32,initializer),
    'd4conv282': tf.get_variable('d4conv282b',[24],tf.float32,initializer),
    'd4conv291': tf.get_variable('d4conv291b',[96],tf.float32,initializer),
    'd4conv292': tf.get_variable('d4conv292b',[24],tf.float32,initializer),
    'd4conv301': tf.get_variable('d4conv301b',[96],tf.float32,initializer),
    'd4conv302': tf.get_variable('d4conv302b',[24],tf.float32,initializer),
    'd4conv311': tf.get_variable('d4conv311b',[96],tf.float32,initializer),
    'd4conv312': tf.get_variable('d4conv312b',[24],tf.float32,initializer),
    'd4conv321': tf.get_variable('d4conv321b',[96],tf.float32,initializer),
    'd4conv322': tf.get_variable('d4conv322b',[24],tf.float32,initializer),

    'fc': tf.get_variable('fcb',[10],tf.float32,initializer),
}

x=tf.placeholder(tf.float32,[100,32,32,3])
y=tf.placeholder(tf.float32)

def conv(x,weight,bias):
    batchnorm=tf.layers.batch_normalization(x)
    activate=tf.nn.relu(batchnorm,name='relu')
    convolution=tf.add(tf.nn.conv2d(activate,weight,strides=[1,1,1,1],padding='SAME'),bias,name='conv')
    return convolution

def subblock(x,weight11,bias11,weight33,bias33):
    conv11=conv(x,weight11,bias11)
    conv33=conv(conv11,weight33,bias33)
    return tf.concat([x,conv33],axis=3)

def avgpool(x,kernel,stride):
    return tf.nn.avg_pool(x,ksize=[1,kernel,kernel,1],strides=[1,stride,stride,1],padding='SAME')

def maxpool(x,kernel,stride):
    return tf.nn.max_pool(x,ksize=[1,kernel,kernel,1],strides=[1,stride,stride,1],padding='SAME')

def transition(x,weight11,bias11):
    conv11=conv(x,weight11,bias11)
    return avgpool(conv11,2,2)

def classification(x,weightfc,biasfc):
    globalpool=tf.reshape(tf.nn.avg_pool(x,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID'),[-1,1245])
    fc=tf.add(tf.matmul(globalpool,weightfc),biasfc)
    return fc

def network(x):
    with tf.name_scope('InputLayer'):
        conv1=conv(x,weights['conv1'],biases['conv1'])
        pool=maxpool(conv1,3,1)

    with tf.name_scope('DenseBock1'):
        d1conv1=subblock(pool,weights['d1conv11'],biases['d1conv11'],weights['d1conv12'],biases['d1conv12'])
        d1conv2=subblock(d1conv1,weights['d1conv21'],biases['d1conv21'],weights['d1conv22'],biases['d1conv22'])
        d1conv3=subblock(d1conv2,weights['d1conv31'],biases['d1conv31'],weights['d1conv32'],biases['d1conv32'])
        d1conv4=subblock(d1conv3,weights['d1conv41'],biases['d1conv41'],weights['d1conv42'],biases['d1conv42'])
        d1conv5=subblock(d1conv4,weights['d1conv51'],biases['d1conv51'],weights['d1conv52'],biases['d1conv52'])
        d1conv6=subblock(d1conv5,weights['d1conv61'],biases['d1conv61'],weights['d1conv62'],biases['d1conv62'])

    with tf.name_scope('Transition1'):
        t1=transition(d1conv6,weights['t1conv'],biases['t1conv'])

    with tf.name_scope('DenseBlock2'):
        d2conv1=subblock(t1,weights['d2conv11'],biases['d2conv11'],weights['d2conv12'],biases['d2conv12'])
        d2conv2=subblock(d2conv1,weights['d2conv21'],biases['d2conv21'],weights['d2conv22'],biases['d2conv22'])
        d2conv3=subblock(d2conv2,weights['d2conv31'],biases['d2conv31'],weights['d2conv32'],biases['d2conv32'])
        d2conv4=subblock(d2conv3,weights['d2conv41'],biases['d2conv41'],weights['d2conv42'],biases['d2conv42'])
        d2conv5=subblock(d2conv4,weights['d2conv51'],biases['d2conv51'],weights['d2conv52'],biases['d2conv52'])
        d2conv6=subblock(d2conv5,weights['d2conv61'],biases['d2conv61'],weights['d2conv62'],biases['d2conv62'])
        d2conv7=subblock(d2conv6,weights['d2conv71'],biases['d2conv71'],weights['d2conv72'],biases['d2conv72'])
        d2conv8=subblock(d2conv7,weights['d2conv81'],biases['d2conv81'],weights['d2conv82'],biases['d2conv82'])
        d2conv9=subblock(d2conv8,weights['d2conv91'],biases['d2conv91'],weights['d2conv92'],biases['d2conv92'])
        d2conv10=subblock(d2conv9,weights['d2conv101'],biases['d2conv101'],weights['d2conv102'],biases['d2conv102'])
        d2conv11=subblock(d2conv10,weights['d2conv111'],biases['d2conv111'],weights['d2conv112'],biases['d2conv112'])
        d2conv12=subblock(d2conv11,weights['d2conv121'],biases['d2conv121'],weights['d2conv122'],biases['d2conv122'])

    with tf.name_scope('Transition2'):
        t2=transition(d2conv12,weights['t2conv'],biases['t2conv'])

    with tf.name_scope('DenseBlock3'):
        d3conv1=subblock(t2,weights['d3conv11'],biases['d3conv11'],weights['d3conv12'],biases['d3conv12'])
        d3conv2=subblock(d3conv1,weights['d3conv21'],biases['d3conv21'],weights['d3conv22'],biases['d3conv22'])
        d3conv3=subblock(d3conv2,weights['d3conv31'],biases['d3conv31'],weights['d3conv32'],biases['d3conv32'])
        d3conv4=subblock(d3conv3,weights['d3conv41'],biases['d3conv41'],weights['d3conv42'],biases['d3conv42'])
        d3conv5=subblock(d3conv4,weights['d3conv51'],biases['d3conv51'],weights['d3conv52'],biases['d3conv52'])
        d3conv6=subblock(d3conv5,weights['d3conv61'],biases['d3conv61'],weights['d3conv62'],biases['d3conv62'])
        d3conv7=subblock(d3conv6,weights['d3conv71'],biases['d3conv71'],weights['d3conv72'],biases['d3conv72'])
        d3conv8=subblock(d3conv7,weights['d3conv81'],biases['d3conv81'],weights['d3conv82'],biases['d3conv82'])
        d3conv9=subblock(d3conv8,weights['d3conv91'],biases['d3conv91'],weights['d3conv92'],biases['d3conv92'])
        d3conv10=subblock(d3conv9,weights['d3conv101'],biases['d3conv101'],weights['d3conv102'],biases['d3conv102'])
        d3conv11=subblock(d3conv10,weights['d3conv111'],biases['d3conv111'],weights['d3conv112'],biases['d3conv112'])
        d3conv12=subblock(d3conv11,weights['d3conv121'],biases['d3conv121'],weights['d3conv122'],biases['d3conv122'])
        d3conv13=subblock(d3conv12,weights['d3conv131'],biases['d3conv131'],weights['d3conv132'],biases['d3conv132'])
        d3conv14=subblock(d3conv13,weights['d3conv141'],biases['d3conv141'],weights['d3conv142'],biases['d3conv142'])
        d3conv15=subblock(d3conv14,weights['d3conv151'],biases['d3conv151'],weights['d3conv152'],biases['d3conv152'])
        d3conv16=subblock(d3conv15,weights['d3conv161'],biases['d3conv161'],weights['d3conv162'],biases['d3conv162'])
        d3conv17=subblock(d3conv16,weights['d3conv171'],biases['d3conv171'],weights['d3conv172'],biases['d3conv172'])
        d3conv18=subblock(d3conv17,weights['d3conv181'],biases['d3conv181'],weights['d3conv182'],biases['d3conv182'])
        d3conv19=subblock(d3conv18,weights['d3conv191'],biases['d3conv191'],weights['d3conv192'],biases['d3conv192'])
        d3conv20=subblock(d3conv19,weights['d3conv201'],biases['d3conv201'],weights['d3conv202'],biases['d3conv202'])
        d3conv21=subblock(d3conv20,weights['d3conv211'],biases['d3conv211'],weights['d3conv212'],biases['d3conv212'])
        d3conv22=subblock(d3conv21,weights['d3conv221'],biases['d3conv221'],weights['d3conv222'],biases['d3conv222'])
        d3conv23=subblock(d3conv22,weights['d3conv231'],biases['d3conv231'],weights['d3conv232'],biases['d3conv232'])
        d3conv24=subblock(d3conv23,weights['d3conv241'],biases['d3conv241'],weights['d3conv242'],biases['d3conv242'])
        d3conv25=subblock(d3conv24,weights['d3conv251'],biases['d3conv251'],weights['d3conv252'],biases['d3conv252'])
        d3conv26=subblock(d3conv25,weights['d3conv261'],biases['d3conv261'],weights['d3conv262'],biases['d3conv262'])
        d3conv27=subblock(d3conv26,weights['d3conv271'],biases['d3conv271'],weights['d3conv272'],biases['d3conv272'])
        d3conv28=subblock(d3conv27,weights['d3conv281'],biases['d3conv281'],weights['d3conv282'],biases['d3conv282'])
        d3conv29=subblock(d3conv28,weights['d3conv291'],biases['d3conv291'],weights['d3conv292'],biases['d3conv292'])
        d3conv30=subblock(d3conv29,weights['d3conv301'],biases['d3conv301'],weights['d3conv302'],biases['d3conv302'])
        d3conv31=subblock(d3conv30,weights['d3conv311'],biases['d3conv311'],weights['d3conv312'],biases['d3conv312'])
        d3conv32=subblock(d3conv31,weights['d3conv321'],biases['d3conv321'],weights['d3conv322'],biases['d3conv322'])

    with tf.name_scope('Transition3'):
        t3=transition(d3conv32,weights['t3conv'],biases['t3conv'])

    with tf.name_scope('DenseBlock4'):
        d4conv1=subblock(t3,weights['d4conv11'],biases['d4conv11'],weights['d4conv12'],biases['d4conv12'])
        d4conv2=subblock(d4conv1,weights['d4conv21'],biases['d4conv21'],weights['d4conv22'],biases['d4conv22'])
        d4conv3=subblock(d4conv2,weights['d4conv31'],biases['d4conv31'],weights['d4conv32'],biases['d4conv32'])
        d4conv4=subblock(d4conv3,weights['d4conv41'],biases['d4conv41'],weights['d4conv42'],biases['d4conv42'])
        d4conv5=subblock(d4conv4,weights['d4conv51'],biases['d4conv51'],weights['d4conv52'],biases['d4conv52'])
        d4conv6=subblock(d4conv5,weights['d4conv61'],biases['d4conv61'],weights['d4conv62'],biases['d4conv62'])
        d4conv7=subblock(d4conv6,weights['d4conv71'],biases['d4conv71'],weights['d4conv72'],biases['d4conv72'])
        d4conv8=subblock(d4conv7,weights['d4conv81'],biases['d4conv81'],weights['d4conv82'],biases['d4conv82'])
        d4conv9=subblock(d4conv8,weights['d4conv91'],biases['d4conv91'],weights['d4conv92'],biases['d4conv92'])
        d4conv10=subblock(d4conv9,weights['d4conv101'],biases['d4conv101'],weights['d4conv102'],biases['d4conv102'])
        d4conv11=subblock(d4conv10,weights['d4conv111'],biases['d4conv111'],weights['d4conv112'],biases['d4conv112'])
        d4conv12=subblock(d4conv11,weights['d4conv121'],biases['d4conv121'],weights['d4conv122'],biases['d4conv122'])
        d4conv13=subblock(d4conv12,weights['d4conv131'],biases['d4conv131'],weights['d4conv132'],biases['d4conv132'])
        d4conv14=subblock(d4conv13,weights['d4conv141'],biases['d4conv141'],weights['d4conv142'],biases['d4conv142'])
        d4conv15=subblock(d4conv14,weights['d4conv151'],biases['d4conv151'],weights['d4conv152'],biases['d4conv152'])
        d4conv16=subblock(d4conv15,weights['d4conv161'],biases['d4conv161'],weights['d4conv162'],biases['d4conv162'])
        d4conv17=subblock(d4conv16,weights['d4conv171'],biases['d4conv171'],weights['d4conv172'],biases['d4conv172'])
        d4conv18=subblock(d4conv17,weights['d4conv181'],biases['d4conv181'],weights['d4conv182'],biases['d4conv182'])
        d4conv19=subblock(d4conv18,weights['d4conv191'],biases['d4conv191'],weights['d4conv192'],biases['d4conv192'])
        d4conv20=subblock(d4conv19,weights['d4conv201'],biases['d4conv201'],weights['d4conv202'],biases['d4conv202'])
        d4conv21=subblock(d4conv20,weights['d4conv211'],biases['d4conv211'],weights['d4conv212'],biases['d4conv212'])
        d4conv22=subblock(d4conv21,weights['d4conv221'],biases['d4conv221'],weights['d4conv222'],biases['d4conv222'])
        d4conv23=subblock(d4conv22,weights['d4conv231'],biases['d4conv231'],weights['d4conv232'],biases['d4conv232'])
        d4conv24=subblock(d4conv23,weights['d4conv241'],biases['d4conv241'],weights['d4conv242'],biases['d4conv242'])
        d4conv25=subblock(d4conv24,weights['d4conv251'],biases['d4conv251'],weights['d4conv252'],biases['d4conv252'])
        d4conv26=subblock(d4conv25,weights['d4conv261'],biases['d4conv261'],weights['d4conv262'],biases['d4conv262'])
        d4conv27=subblock(d4conv26,weights['d4conv271'],biases['d4conv271'],weights['d4conv272'],biases['d4conv272'])
        d4conv28=subblock(d4conv27,weights['d4conv281'],biases['d4conv281'],weights['d4conv282'],biases['d4conv282'])
        d4conv29=subblock(d4conv28,weights['d4conv291'],biases['d4conv291'],weights['d4conv292'],biases['d4conv292'])
        d4conv30=subblock(d4conv29,weights['d4conv301'],biases['d4conv301'],weights['d4conv302'],biases['d4conv302'])
        d4conv31=subblock(d4conv30,weights['d4conv311'],biases['d4conv311'],weights['d4conv312'],biases['d4conv312'])
        d4conv32=subblock(d4conv31,weights['d4conv321'],biases['d4conv321'],weights['d4conv322'],biases['d4conv322'])

    with tf.name_scope('ClassificationLayer'):
        out=classification(d4conv32,weights['fc'],biases['fc'])

    return out

predict_y=network(x)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimize=tf.train.AdamOptimizer().minimize(loss)

def save(path,sess):
    saver=tf.train.Saver()
    saver.save(sess,path)

def restore(sess):
    saver=tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint('./Dense-169-24/'))


def train():
    print('Loading train images')
    path='train/'
    files=os.listdir(path)
    images=[]
    for file in files: images.append(misc.imread(path+file))
    print('Images loaded')
    labels=[]
    with open('trainLabels.csv','rt') as csvfile:
        labelcsv=csv.reader(csvfile,delimiter=' ',quotechar='|')
        for row in labelcsv:
            row=row[0].rstrip()
            row=row.split(',')
            row=row[-1]
            if row=='airplane': labels.append([1,0,0,0,0,0,0,0,0,0])
            if row=='automobile': labels.append([0,1,0,0,0,0,0,0,0,0])
            if row=='bird': labels.append([0,0,1,0,0,0,0,0,0,0])
            if row=='cat': labels.append([0,0,0,1,0,0,0,0,0,0])
            if row=='deer': labels.append([0,0,0,0,1,0,0,0,0,0])
            if row=='dog': labels.append([0,0,0,0,0,1,0,0,0,0])
            if row=='frog': labels.append([0,0,0,0,0,0,1,0,0,0])
            if row=='horse': labels.append([0,0,0,0,0,0,0,1,0,0])
            if row=='ship': labels.append([0,0,0,0,0,0,0,0,1,0])
            if row=='truck': labels.append([0,0,0,0,0,0,0,0,0,1])
    print('Labels loaded')
    print('Batch size : 10')
    print('Variables saved after every 2 Epochs')
    print('100 Epochs')
    with tf.Session() as sess:
        try: restore(sess)
        except: print("Couldn't restore, Reinitializing"); sess.run(tf.global_variables_initializer())
        for epoch in range(1,101):
            epochloss=0
            for i in range(1,501):
                batchimages,batchlabels=images[100*(i-1):100*(i)],labels[100*(i-1):100*i]
                batchloss,_=sess.run([loss,optimize],feed_dict={x:batchimages, y:batchlabels})
                print('Batch',i,'out of 500 completed in epoch',epoch,'. Batch loss : ', batchloss)
                epochloss+=batchloss
            print('Epoch',epoch,'completed, loss :',epochloss)
            if epoch%2==0: save('Dense-169-24/var.ckpt',sess)
        print('Network trained')

train()
