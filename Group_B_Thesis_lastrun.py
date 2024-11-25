#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on November 24, 2024, at 00:51
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'untitled'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 864]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Semester_7\\Thesis\\Thesis_exp_main\\thesis-main\\Group_B_Thesis_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    if deviceManager.getDevice('key_resp_p') is None:
        # initialise key_resp_p
        key_resp_p = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_p',
        )
    if deviceManager.getDevice('key_resp_11') is None:
        # initialise key_resp_11
        key_resp_11 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_11',
        )
    if deviceManager.getDevice('key_resp_10') is None:
        # initialise key_resp_10
        key_resp_10 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_10',
        )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    if deviceManager.getDevice('key_resp_8') is None:
        # initialise key_resp_8
        key_resp_8 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_8',
        )
    if deviceManager.getDevice('key_resp_9') is None:
        # initialise key_resp_9
        key_resp_9 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_9',
        )
    if deviceManager.getDevice('key_resp_6_p') is None:
        # initialise key_resp_6_p
        key_resp_6_p = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6_p',
        )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Intruduction" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text="Welcome to the experiment. This is a research experiment for my undergraduate thesis.\n\npress 'spacebar' to continue.\n\n\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "self_other_instructions" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text="In this study, you will be shown two shapes (square and triangle) with two labels (you and stranger). You have to memorize this association. \n\n- You are represented by a Square (You-Square)\n-There is a Stranger (an unknown person to you), who is represented by Triangle (Stranger-Triangle)\n\nTake a few minutes to memorize this association. Let me know when you have memorized and ready to proceed to the experiment.\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "self_other_info" ---
    square_image = visual.ImageStim(
        win=win,
        name='square_image', 
        image='square.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, 0), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    triangle_image = visual.ImageStim(
        win=win,
        name='triangle_image', 
        image='triangle.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, 0), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    self_label = visual.TextStim(win=win, name='self_label',
        text='stranger',
        font='Arial',
        pos=(-0.3, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    other_label = visual.TextStim(win=win, name='other_label',
        text='you',
        font='Arial',
        pos=(0.3, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    text_5 = visual.TextStim(win=win, name='text_5',
        text="press 'spacebar' to continue",
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "self_other_starting_2" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text="Your task is to judge whether the shape label pairing is match or non match. If the shape label pairing is match press 'm' and if its non match press 'z'.\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    
    # --- Initialize components for Routine "b1" ---
    p_fication = visual.ShapeStim(
        win=win, name='p_fication', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    p_fixation_3 = visual.ShapeStim(
        win=win, name='p_fixation_3', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_text = visual.TextStim(win=win, name='p_text',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_p = keyboard.Keyboard(deviceName='key_resp_p')
    p_image = visual.ImageStim(
        win=win,
        name='p_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "b_feedback" ---
    text_15 = visual.TextStim(win=win, name='text_15',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "start2" ---
    start_2 = visual.TextStim(win=win, name='start_2',
        text="Press 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    
    # --- Initialize components for Routine "b1" ---
    p_fication = visual.ShapeStim(
        win=win, name='p_fication', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    p_fixation_3 = visual.ShapeStim(
        win=win, name='p_fixation_3', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_text = visual.TextStim(win=win, name='p_text',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_p = keyboard.Keyboard(deviceName='key_resp_p')
    p_image = visual.ImageStim(
        win=win,
        name='p_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "b_feedback" ---
    text_15 = visual.TextStim(win=win, name='text_15',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "break_3" ---
    text_13 = visual.TextStim(win=win, name='text_13',
        text='Now you will have a break of 2 minutes.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_14 = visual.TextStim(win=win, name='text_14',
        text='Break is over press the spacebar to continue the experiment',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_10 = keyboard.Keyboard(deviceName='key_resp_10')
    
    # --- Initialize components for Routine "self_instructions_2" ---
    text_10 = visual.TextStim(win=win, name='text_10',
        text="In this study, you will be shown two shapes (circle and diamond) with two labels (disgusting self and desirable self). You have to memorize this association. \n\n- Disgusting self is represented by a Circle (Disgusting self-Circle) and\n-Desirable self is represented by Diamond (Desirable self-Diamond)\n\nTake a few minutes to memorize this association. Let me know when you have memorized and ready to proceed to the experiment.\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    
    # --- Initialize components for Routine "self_intro_2" ---
    circle_image = visual.ImageStim(
        win=win,
        name='circle_image', 
        image='circle.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, 0), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    diamond_image = visual.ImageStim(
        win=win,
        name='diamond_image', 
        image='diamond.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, 0), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    Disgusting_self_lable = visual.TextStim(win=win, name='Disgusting_self_lable',
        text='disgusting self',
        font='Arial',
        pos=(0.3, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    Desirable_self_lable = visual.TextStim(win=win, name='Desirable_self_lable',
        text='desirable self',
        font='Arial',
        pos=(-0.3, -0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_8 = keyboard.Keyboard(deviceName='key_resp_8')
    text_11 = visual.TextStim(win=win, name='text_11',
        text="press 'spacebar' to continue",
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "self_starting_3" ---
    text_12 = visual.TextStim(win=win, name='text_12',
        text="Your task is to judge whether the shape label pairing is match or non match. If the shape label pairing is match press 'm' and if its non match press 'z'.\n\npress 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_9 = keyboard.Keyboard(deviceName='key_resp_9')
    
    # --- Initialize components for Routine "practice2" ---
    p_fixation_2 = visual.ShapeStim(
        win=win, name='p_fixation_2', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    p_fixation_4 = visual.ShapeStim(
        win=win, name='p_fixation_4', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    p_text_8 = visual.TextStim(win=win, name='p_text_8',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_6_p = keyboard.Keyboard(deviceName='key_resp_6_p')
    p_image_2 = visual.ImageStim(
        win=win,
        name='p_image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "feedback_2_p" ---
    text_16 = visual.TextStim(win=win, name='text_16',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "start2" ---
    start_2 = visual.TextStim(win=win, name='start_2',
        text="Press 'spacebar' to continue",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard(deviceName='key_resp_11')
    
    # --- Initialize components for Routine "trial_2" ---
    fixation_2 = visual.ShapeStim(
        win=win, name='fixation_2', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    fixation_4 = visual.ShapeStim(
        win=win, name='fixation_4', vertices='cross',
        size=(0.03, 0.03),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    text_8 = visual.TextStim(win=win, name='text_8',
        text='',
        font='Open Sans',
        pos=(0, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.2), draggable=False, size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "feedback_2" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Thank_you" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='Thank_You',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Intruduction" ---
    # create an object to store info about Routine Intruduction
    Intruduction = data.Routine(
        name='Intruduction',
        components=[text_3, key_resp_2],
    )
    Intruduction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for Intruduction
    Intruduction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intruduction.tStart = globalClock.getTime(format='float')
    Intruduction.status = STARTED
    thisExp.addData('Intruduction.started', Intruduction.tStart)
    Intruduction.maxDuration = None
    # keep track of which components have finished
    IntruductionComponents = Intruduction.components
    for thisComponent in Intruduction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Intruduction" ---
    Intruduction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Intruduction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Intruduction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intruduction" ---
    for thisComponent in Intruduction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intruduction
    Intruduction.tStop = globalClock.getTime(format='float')
    Intruduction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intruduction.stopped', Intruduction.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Intruduction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_other_instructions" ---
    # create an object to store info about Routine self_other_instructions
    self_other_instructions = data.Routine(
        name='self_other_instructions',
        components=[text_4, key_resp_3],
    )
    self_other_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for self_other_instructions
    self_other_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_other_instructions.tStart = globalClock.getTime(format='float')
    self_other_instructions.status = STARTED
    thisExp.addData('self_other_instructions.started', self_other_instructions.tStart)
    self_other_instructions.maxDuration = None
    # keep track of which components have finished
    self_other_instructionsComponents = self_other_instructions.components
    for thisComponent in self_other_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_other_instructions" ---
    self_other_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_other_instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_other_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_other_instructions" ---
    for thisComponent in self_other_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_other_instructions
    self_other_instructions.tStop = globalClock.getTime(format='float')
    self_other_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_other_instructions.stopped', self_other_instructions.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "self_other_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_other_info" ---
    # create an object to store info about Routine self_other_info
    self_other_info = data.Routine(
        name='self_other_info',
        components=[square_image, triangle_image, self_label, other_label, key_resp_4, text_5],
    )
    self_other_info.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # store start times for self_other_info
    self_other_info.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_other_info.tStart = globalClock.getTime(format='float')
    self_other_info.status = STARTED
    thisExp.addData('self_other_info.started', self_other_info.tStart)
    self_other_info.maxDuration = None
    # keep track of which components have finished
    self_other_infoComponents = self_other_info.components
    for thisComponent in self_other_info.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_other_info" ---
    self_other_info.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *square_image* updates
        
        # if square_image is starting this frame...
        if square_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_image.frameNStart = frameN  # exact frame index
            square_image.tStart = t  # local t and not account for scr refresh
            square_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_image.started')
            # update status
            square_image.status = STARTED
            square_image.setAutoDraw(True)
        
        # if square_image is active this frame...
        if square_image.status == STARTED:
            # update params
            pass
        
        # *triangle_image* updates
        
        # if triangle_image is starting this frame...
        if triangle_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            triangle_image.frameNStart = frameN  # exact frame index
            triangle_image.tStart = t  # local t and not account for scr refresh
            triangle_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(triangle_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'triangle_image.started')
            # update status
            triangle_image.status = STARTED
            triangle_image.setAutoDraw(True)
        
        # if triangle_image is active this frame...
        if triangle_image.status == STARTED:
            # update params
            pass
        
        # *self_label* updates
        
        # if self_label is starting this frame...
        if self_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            self_label.frameNStart = frameN  # exact frame index
            self_label.tStart = t  # local t and not account for scr refresh
            self_label.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(self_label, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'self_label.started')
            # update status
            self_label.status = STARTED
            self_label.setAutoDraw(True)
        
        # if self_label is active this frame...
        if self_label.status == STARTED:
            # update params
            pass
        
        # *other_label* updates
        
        # if other_label is starting this frame...
        if other_label.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            other_label.frameNStart = frameN  # exact frame index
            other_label.tStart = t  # local t and not account for scr refresh
            other_label.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(other_label, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'other_label.started')
            # update status
            other_label.status = STARTED
            other_label.setAutoDraw(True)
        
        # if other_label is active this frame...
        if other_label.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_other_info.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_other_info.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_other_info" ---
    for thisComponent in self_other_info.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_other_info
    self_other_info.tStop = globalClock.getTime(format='float')
    self_other_info.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_other_info.stopped', self_other_info.tStop)
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "self_other_info" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_other_starting_2" ---
    # create an object to store info about Routine self_other_starting_2
    self_other_starting_2 = data.Routine(
        name='self_other_starting_2',
        components=[text_6, key_resp_5],
    )
    self_other_starting_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_5
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # store start times for self_other_starting_2
    self_other_starting_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_other_starting_2.tStart = globalClock.getTime(format='float')
    self_other_starting_2.status = STARTED
    thisExp.addData('self_other_starting_2.started', self_other_starting_2.tStart)
    self_other_starting_2.maxDuration = None
    # keep track of which components have finished
    self_other_starting_2Components = self_other_starting_2.components
    for thisComponent in self_other_starting_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_other_starting_2" ---
    self_other_starting_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_6.started')
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_5.started')
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_other_starting_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_other_starting_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_other_starting_2" ---
    for thisComponent in self_other_starting_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_other_starting_2
    self_other_starting_2.tStop = globalClock.getTime(format='float')
    self_other_starting_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_other_starting_2.stopped', self_other_starting_2.tStop)
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    thisExp.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        thisExp.addData('key_resp_5.rt', key_resp_5.rt)
        thisExp.addData('key_resp_5.duration', key_resp_5.duration)
    thisExp.nextEntry()
    # the Routine "self_other_starting_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    p_trials = data.TrialHandler2(
        name='p_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('B_block1.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(p_trials)  # add the loop to the experiment
    thisP_trial = p_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisP_trial.rgb)
    if thisP_trial != None:
        for paramName in thisP_trial:
            globals()[paramName] = thisP_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisP_trial in p_trials:
        currentLoop = p_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisP_trial.rgb)
        if thisP_trial != None:
            for paramName in thisP_trial:
                globals()[paramName] = thisP_trial[paramName]
        
        # --- Prepare to start Routine "b1" ---
        # create an object to store info about Routine b1
        b1 = data.Routine(
            name='b1',
            components=[p_fication, p_fixation_3, p_text, key_resp_p, p_image],
        )
        b1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        p_text.setText(label_b)
        # create starting attributes for key_resp_p
        key_resp_p.keys = []
        key_resp_p.rt = []
        _key_resp_p_allKeys = []
        p_image.setImage(stimulus_b)
        # store start times for b1
        b1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        b1.tStart = globalClock.getTime(format='float')
        b1.status = STARTED
        thisExp.addData('b1.started', b1.tStart)
        b1.maxDuration = None
        # keep track of which components have finished
        b1Components = b1.components
        for thisComponent in b1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "b1" ---
        # if trial has changed, end Routine now
        if isinstance(p_trials, data.TrialHandler2) and thisP_trial.thisN != p_trials.thisTrial.thisN:
            continueRoutine = False
        b1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *p_fication* updates
            
            # if p_fication is starting this frame...
            if p_fication.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fication.frameNStart = frameN  # exact frame index
                p_fication.tStart = t  # local t and not account for scr refresh
                p_fication.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fication, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fication.started')
                # update status
                p_fication.status = STARTED
                p_fication.setAutoDraw(True)
            
            # if p_fication is active this frame...
            if p_fication.status == STARTED:
                # update params
                pass
            
            # if p_fication is stopping this frame...
            if p_fication.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fication.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fication.tStop = t  # not accounting for scr refresh
                    p_fication.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fication.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fication.stopped')
                    # update status
                    p_fication.status = FINISHED
                    p_fication.setAutoDraw(False)
            
            # *p_fixation_3* updates
            
            # if p_fixation_3 is starting this frame...
            if p_fixation_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_3.frameNStart = frameN  # exact frame index
                p_fixation_3.tStart = t  # local t and not account for scr refresh
                p_fixation_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_3.started')
                # update status
                p_fixation_3.status = STARTED
                p_fixation_3.setAutoDraw(True)
            
            # if p_fixation_3 is active this frame...
            if p_fixation_3.status == STARTED:
                # update params
                pass
            
            # if p_fixation_3 is stopping this frame...
            if p_fixation_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fixation_3.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fixation_3.tStop = t  # not accounting for scr refresh
                    p_fixation_3.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fixation_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fixation_3.stopped')
                    # update status
                    p_fixation_3.status = FINISHED
                    p_fixation_3.setAutoDraw(False)
            
            # *p_text* updates
            
            # if p_text is starting this frame...
            if p_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_text.frameNStart = frameN  # exact frame index
                p_text.tStart = t  # local t and not account for scr refresh
                p_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_text.started')
                # update status
                p_text.status = STARTED
                p_text.setAutoDraw(True)
            
            # if p_text is active this frame...
            if p_text.status == STARTED:
                # update params
                pass
            
            # if p_text is stopping this frame...
            if p_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_text.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_text.tStop = t  # not accounting for scr refresh
                    p_text.tStopRefresh = tThisFlipGlobal  # on global time
                    p_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_text.stopped')
                    # update status
                    p_text.status = FINISHED
                    p_text.setAutoDraw(False)
            
            # *key_resp_p* updates
            waitOnFlip = False
            
            # if key_resp_p is starting this frame...
            if key_resp_p.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_p.frameNStart = frameN  # exact frame index
                key_resp_p.tStart = t  # local t and not account for scr refresh
                key_resp_p.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_p, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_p.started')
                # update status
                key_resp_p.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_p.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_p.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_p is stopping this frame...
            if key_resp_p.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_p.tStartRefresh + 1.1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_p.tStop = t  # not accounting for scr refresh
                    key_resp_p.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_p.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_p.stopped')
                    # update status
                    key_resp_p.status = FINISHED
                    key_resp_p.status = FINISHED
            if key_resp_p.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_p.getKeys(keyList=["m","z"], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_p_allKeys.extend(theseKeys)
                if len(_key_resp_p_allKeys):
                    key_resp_p.keys = _key_resp_p_allKeys[-1].name  # just the last key pressed
                    key_resp_p.rt = _key_resp_p_allKeys[-1].rt
                    key_resp_p.duration = _key_resp_p_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *p_image* updates
            
            # if p_image is starting this frame...
            if p_image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_image.frameNStart = frameN  # exact frame index
                p_image.tStart = t  # local t and not account for scr refresh
                p_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_image.started')
                # update status
                p_image.status = STARTED
                p_image.setAutoDraw(True)
            
            # if p_image is active this frame...
            if p_image.status == STARTED:
                # update params
                pass
            
            # if p_image is stopping this frame...
            if p_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_image.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_image.tStop = t  # not accounting for scr refresh
                    p_image.tStopRefresh = tThisFlipGlobal  # on global time
                    p_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_image.stopped')
                    # update status
                    p_image.status = FINISHED
                    p_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                b1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in b1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "b1" ---
        for thisComponent in b1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for b1
        b1.tStop = globalClock.getTime(format='float')
        b1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('b1.stopped', b1.tStop)
        # check responses
        if key_resp_p.keys in ['', [], None]:  # No response was made
            key_resp_p.keys = None
        p_trials.addData('key_resp_p.keys',key_resp_p.keys)
        if key_resp_p.keys != None:  # we had a response
            p_trials.addData('key_resp_p.rt', key_resp_p.rt)
            p_trials.addData('key_resp_p.duration', key_resp_p.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if b1.maxDurationReached:
            routineTimer.addTime(-b1.maxDuration)
        elif b1.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.600000)
        
        # --- Prepare to start Routine "b_feedback" ---
        # create an object to store info about Routine b_feedback
        b_feedback = data.Routine(
            name='b_feedback',
            components=[text_15],
        )
        b_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        # Check if they responded correctly, incorrectly, or missed
        is_match1 = int(is_match_b)  # is_match_b is now treated as a variable from the Excel file
        
        if not key_resp_p.keys:  # No response was made
            feedback_message = 'Miss'
        elif key_resp_p.keys == 'm' and is_match1:  # Correct match response
            feedback_message = 'Correct'
        elif key_resp_p.keys == 'z' and not is_match1:  # Correct not-match response
            feedback_message = 'Correct'
        else:  # Any other response is incorrect
            feedback_message = 'Incorrect'
           
        
        
        text_15.setText(feedback_message)
        # store start times for b_feedback
        b_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        b_feedback.tStart = globalClock.getTime(format='float')
        b_feedback.status = STARTED
        thisExp.addData('b_feedback.started', b_feedback.tStart)
        b_feedback.maxDuration = None
        # keep track of which components have finished
        b_feedbackComponents = b_feedback.components
        for thisComponent in b_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "b_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(p_trials, data.TrialHandler2) and thisP_trial.thisN != p_trials.thisTrial.thisN:
            continueRoutine = False
        b_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_15* updates
            
            # if text_15 is starting this frame...
            if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_15.frameNStart = frameN  # exact frame index
                text_15.tStart = t  # local t and not account for scr refresh
                text_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_15.started')
                # update status
                text_15.status = STARTED
                text_15.setAutoDraw(True)
            
            # if text_15 is active this frame...
            if text_15.status == STARTED:
                # update params
                pass
            
            # if text_15 is stopping this frame...
            if text_15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_15.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_15.tStop = t  # not accounting for scr refresh
                    text_15.tStopRefresh = tThisFlipGlobal  # on global time
                    text_15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_15.stopped')
                    # update status
                    text_15.status = FINISHED
                    text_15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                b_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in b_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "b_feedback" ---
        for thisComponent in b_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for b_feedback
        b_feedback.tStop = globalClock.getTime(format='float')
        b_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('b_feedback.stopped', b_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if b_feedback.maxDurationReached:
            routineTimer.addTime(-b_feedback.maxDuration)
        elif b_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'p_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start2" ---
    # create an object to store info about Routine start2
    start2 = data.Routine(
        name='start2',
        components=[start_2, key_resp_11],
    )
    start2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_11
    key_resp_11.keys = []
    key_resp_11.rt = []
    _key_resp_11_allKeys = []
    # store start times for start2
    start2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start2.tStart = globalClock.getTime(format='float')
    start2.status = STARTED
    thisExp.addData('start2.started', start2.tStart)
    start2.maxDuration = None
    # keep track of which components have finished
    start2Components = start2.components
    for thisComponent in start2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start2" ---
    start2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_2* updates
        
        # if start_2 is starting this frame...
        if start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_2.frameNStart = frameN  # exact frame index
            start_2.tStart = t  # local t and not account for scr refresh
            start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_2.started')
            # update status
            start_2.status = STARTED
            start_2.setAutoDraw(True)
        
        # if start_2 is active this frame...
        if start_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_11* updates
        waitOnFlip = False
        
        # if key_resp_11 is starting this frame...
        if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_11.frameNStart = frameN  # exact frame index
            key_resp_11.tStart = t  # local t and not account for scr refresh
            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_11.started')
            # update status
            key_resp_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_11.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_11_allKeys.extend(theseKeys)
            if len(_key_resp_11_allKeys):
                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start2" ---
    for thisComponent in start2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start2
    start2.tStop = globalClock.getTime(format='float')
    start2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start2.stopped', start2.tStop)
    # check responses
    if key_resp_11.keys in ['', [], None]:  # No response was made
        key_resp_11.keys = None
    thisExp.addData('key_resp_11.keys',key_resp_11.keys)
    if key_resp_11.keys != None:  # we had a response
        thisExp.addData('key_resp_11.rt', key_resp_11.rt)
        thisExp.addData('key_resp_11.duration', key_resp_11.duration)
    thisExp.nextEntry()
    # the Routine "start2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('B_block1.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "b1" ---
        # create an object to store info about Routine b1
        b1 = data.Routine(
            name='b1',
            components=[p_fication, p_fixation_3, p_text, key_resp_p, p_image],
        )
        b1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        p_text.setText(label_b)
        # create starting attributes for key_resp_p
        key_resp_p.keys = []
        key_resp_p.rt = []
        _key_resp_p_allKeys = []
        p_image.setImage(stimulus_b)
        # store start times for b1
        b1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        b1.tStart = globalClock.getTime(format='float')
        b1.status = STARTED
        thisExp.addData('b1.started', b1.tStart)
        b1.maxDuration = None
        # keep track of which components have finished
        b1Components = b1.components
        for thisComponent in b1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "b1" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        b1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *p_fication* updates
            
            # if p_fication is starting this frame...
            if p_fication.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fication.frameNStart = frameN  # exact frame index
                p_fication.tStart = t  # local t and not account for scr refresh
                p_fication.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fication, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fication.started')
                # update status
                p_fication.status = STARTED
                p_fication.setAutoDraw(True)
            
            # if p_fication is active this frame...
            if p_fication.status == STARTED:
                # update params
                pass
            
            # if p_fication is stopping this frame...
            if p_fication.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fication.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fication.tStop = t  # not accounting for scr refresh
                    p_fication.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fication.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fication.stopped')
                    # update status
                    p_fication.status = FINISHED
                    p_fication.setAutoDraw(False)
            
            # *p_fixation_3* updates
            
            # if p_fixation_3 is starting this frame...
            if p_fixation_3.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_3.frameNStart = frameN  # exact frame index
                p_fixation_3.tStart = t  # local t and not account for scr refresh
                p_fixation_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_3.started')
                # update status
                p_fixation_3.status = STARTED
                p_fixation_3.setAutoDraw(True)
            
            # if p_fixation_3 is active this frame...
            if p_fixation_3.status == STARTED:
                # update params
                pass
            
            # if p_fixation_3 is stopping this frame...
            if p_fixation_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fixation_3.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fixation_3.tStop = t  # not accounting for scr refresh
                    p_fixation_3.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fixation_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fixation_3.stopped')
                    # update status
                    p_fixation_3.status = FINISHED
                    p_fixation_3.setAutoDraw(False)
            
            # *p_text* updates
            
            # if p_text is starting this frame...
            if p_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_text.frameNStart = frameN  # exact frame index
                p_text.tStart = t  # local t and not account for scr refresh
                p_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_text.started')
                # update status
                p_text.status = STARTED
                p_text.setAutoDraw(True)
            
            # if p_text is active this frame...
            if p_text.status == STARTED:
                # update params
                pass
            
            # if p_text is stopping this frame...
            if p_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_text.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_text.tStop = t  # not accounting for scr refresh
                    p_text.tStopRefresh = tThisFlipGlobal  # on global time
                    p_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_text.stopped')
                    # update status
                    p_text.status = FINISHED
                    p_text.setAutoDraw(False)
            
            # *key_resp_p* updates
            waitOnFlip = False
            
            # if key_resp_p is starting this frame...
            if key_resp_p.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_p.frameNStart = frameN  # exact frame index
                key_resp_p.tStart = t  # local t and not account for scr refresh
                key_resp_p.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_p, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_p.started')
                # update status
                key_resp_p.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_p.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_p.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_p is stopping this frame...
            if key_resp_p.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_p.tStartRefresh + 1.1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_p.tStop = t  # not accounting for scr refresh
                    key_resp_p.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_p.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_p.stopped')
                    # update status
                    key_resp_p.status = FINISHED
                    key_resp_p.status = FINISHED
            if key_resp_p.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_p.getKeys(keyList=["m","z"], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_p_allKeys.extend(theseKeys)
                if len(_key_resp_p_allKeys):
                    key_resp_p.keys = _key_resp_p_allKeys[-1].name  # just the last key pressed
                    key_resp_p.rt = _key_resp_p_allKeys[-1].rt
                    key_resp_p.duration = _key_resp_p_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *p_image* updates
            
            # if p_image is starting this frame...
            if p_image.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_image.frameNStart = frameN  # exact frame index
                p_image.tStart = t  # local t and not account for scr refresh
                p_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_image.started')
                # update status
                p_image.status = STARTED
                p_image.setAutoDraw(True)
            
            # if p_image is active this frame...
            if p_image.status == STARTED:
                # update params
                pass
            
            # if p_image is stopping this frame...
            if p_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_image.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_image.tStop = t  # not accounting for scr refresh
                    p_image.tStopRefresh = tThisFlipGlobal  # on global time
                    p_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_image.stopped')
                    # update status
                    p_image.status = FINISHED
                    p_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                b1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in b1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "b1" ---
        for thisComponent in b1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for b1
        b1.tStop = globalClock.getTime(format='float')
        b1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('b1.stopped', b1.tStop)
        # check responses
        if key_resp_p.keys in ['', [], None]:  # No response was made
            key_resp_p.keys = None
        trials.addData('key_resp_p.keys',key_resp_p.keys)
        if key_resp_p.keys != None:  # we had a response
            trials.addData('key_resp_p.rt', key_resp_p.rt)
            trials.addData('key_resp_p.duration', key_resp_p.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if b1.maxDurationReached:
            routineTimer.addTime(-b1.maxDuration)
        elif b1.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.600000)
        
        # --- Prepare to start Routine "b_feedback" ---
        # create an object to store info about Routine b_feedback
        b_feedback = data.Routine(
            name='b_feedback',
            components=[text_15],
        )
        b_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        # Check if they responded correctly, incorrectly, or missed
        is_match1 = int(is_match_b)  # is_match_b is now treated as a variable from the Excel file
        
        if not key_resp_p.keys:  # No response was made
            feedback_message = 'Miss'
        elif key_resp_p.keys == 'm' and is_match1:  # Correct match response
            feedback_message = 'Correct'
        elif key_resp_p.keys == 'z' and not is_match1:  # Correct not-match response
            feedback_message = 'Correct'
        else:  # Any other response is incorrect
            feedback_message = 'Incorrect'
           
        
        
        text_15.setText(feedback_message)
        # store start times for b_feedback
        b_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        b_feedback.tStart = globalClock.getTime(format='float')
        b_feedback.status = STARTED
        thisExp.addData('b_feedback.started', b_feedback.tStart)
        b_feedback.maxDuration = None
        # keep track of which components have finished
        b_feedbackComponents = b_feedback.components
        for thisComponent in b_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "b_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        b_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_15* updates
            
            # if text_15 is starting this frame...
            if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_15.frameNStart = frameN  # exact frame index
                text_15.tStart = t  # local t and not account for scr refresh
                text_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_15.started')
                # update status
                text_15.status = STARTED
                text_15.setAutoDraw(True)
            
            # if text_15 is active this frame...
            if text_15.status == STARTED:
                # update params
                pass
            
            # if text_15 is stopping this frame...
            if text_15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_15.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_15.tStop = t  # not accounting for scr refresh
                    text_15.tStopRefresh = tThisFlipGlobal  # on global time
                    text_15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_15.stopped')
                    # update status
                    text_15.status = FINISHED
                    text_15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                b_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in b_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "b_feedback" ---
        for thisComponent in b_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for b_feedback
        b_feedback.tStop = globalClock.getTime(format='float')
        b_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('b_feedback.stopped', b_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if b_feedback.maxDurationReached:
            routineTimer.addTime(-b_feedback.maxDuration)
        elif b_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "break_3" ---
    # create an object to store info about Routine break_3
    break_3 = data.Routine(
        name='break_3',
        components=[text_13, text_14, key_resp_10],
    )
    break_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_10
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # store start times for break_3
    break_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    break_3.tStart = globalClock.getTime(format='float')
    break_3.status = STARTED
    thisExp.addData('break_3.started', break_3.tStart)
    break_3.maxDuration = None
    # keep track of which components have finished
    break_3Components = break_3.components
    for thisComponent in break_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "break_3" ---
    break_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 6.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_13* updates
        
        # if text_13 is starting this frame...
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_13.started')
            # update status
            text_13.status = STARTED
            text_13.setAutoDraw(True)
        
        # if text_13 is active this frame...
        if text_13.status == STARTED:
            # update params
            pass
        
        # if text_13 is stopping this frame...
        if text_13.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_13.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_13.tStop = t  # not accounting for scr refresh
                text_13.tStopRefresh = tThisFlipGlobal  # on global time
                text_13.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_13.stopped')
                # update status
                text_13.status = FINISHED
                text_13.setAutoDraw(False)
        
        # *text_14* updates
        
        # if text_14 is starting this frame...
        if text_14.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            text_14.frameNStart = frameN  # exact frame index
            text_14.tStart = t  # local t and not account for scr refresh
            text_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_14.started')
            # update status
            text_14.status = STARTED
            text_14.setAutoDraw(True)
        
        # if text_14 is active this frame...
        if text_14.status == STARTED:
            # update params
            pass
        
        # if text_14 is stopping this frame...
        if text_14.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_14.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_14.tStop = t  # not accounting for scr refresh
                text_14.tStopRefresh = tThisFlipGlobal  # on global time
                text_14.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_14.stopped')
                # update status
                text_14.status = FINISHED
                text_14.setAutoDraw(False)
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_10.started')
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_10 is stopping this frame...
        if key_resp_10.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_10.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_10.tStop = t  # not accounting for scr refresh
                key_resp_10.tStopRefresh = tThisFlipGlobal  # on global time
                key_resp_10.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_10.stopped')
                # update status
                key_resp_10.status = FINISHED
                key_resp_10.status = FINISHED
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_3" ---
    for thisComponent in break_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for break_3
    break_3.tStop = globalClock.getTime(format='float')
    break_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('break_3.stopped', break_3.tStop)
    # check responses
    if key_resp_10.keys in ['', [], None]:  # No response was made
        key_resp_10.keys = None
    thisExp.addData('key_resp_10.keys',key_resp_10.keys)
    if key_resp_10.keys != None:  # we had a response
        thisExp.addData('key_resp_10.rt', key_resp_10.rt)
        thisExp.addData('key_resp_10.duration', key_resp_10.duration)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if break_3.maxDurationReached:
        routineTimer.addTime(-break_3.maxDuration)
    elif break_3.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-6.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "self_instructions_2" ---
    # create an object to store info about Routine self_instructions_2
    self_instructions_2 = data.Routine(
        name='self_instructions_2',
        components=[text_10, key_resp_7],
    )
    self_instructions_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # store start times for self_instructions_2
    self_instructions_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_instructions_2.tStart = globalClock.getTime(format='float')
    self_instructions_2.status = STARTED
    thisExp.addData('self_instructions_2.started', self_instructions_2.tStart)
    self_instructions_2.maxDuration = None
    # keep track of which components have finished
    self_instructions_2Components = self_instructions_2.components
    for thisComponent in self_instructions_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_instructions_2" ---
    self_instructions_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_10* updates
        
        # if text_10 is starting this frame...
        if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_10.frameNStart = frameN  # exact frame index
            text_10.tStart = t  # local t and not account for scr refresh
            text_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_10.started')
            # update status
            text_10.status = STARTED
            text_10.setAutoDraw(True)
        
        # if text_10 is active this frame...
        if text_10.status == STARTED:
            # update params
            pass
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_instructions_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_instructions_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_instructions_2" ---
    for thisComponent in self_instructions_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_instructions_2
    self_instructions_2.tStop = globalClock.getTime(format='float')
    self_instructions_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_instructions_2.stopped', self_instructions_2.tStop)
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "self_instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_intro_2" ---
    # create an object to store info about Routine self_intro_2
    self_intro_2 = data.Routine(
        name='self_intro_2',
        components=[circle_image, diamond_image, Disgusting_self_lable, Desirable_self_lable, key_resp_8, text_11],
    )
    self_intro_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_8
    key_resp_8.keys = []
    key_resp_8.rt = []
    _key_resp_8_allKeys = []
    # store start times for self_intro_2
    self_intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_intro_2.tStart = globalClock.getTime(format='float')
    self_intro_2.status = STARTED
    thisExp.addData('self_intro_2.started', self_intro_2.tStart)
    self_intro_2.maxDuration = None
    # keep track of which components have finished
    self_intro_2Components = self_intro_2.components
    for thisComponent in self_intro_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_intro_2" ---
    self_intro_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_image* updates
        
        # if circle_image is starting this frame...
        if circle_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_image.frameNStart = frameN  # exact frame index
            circle_image.tStart = t  # local t and not account for scr refresh
            circle_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_image.started')
            # update status
            circle_image.status = STARTED
            circle_image.setAutoDraw(True)
        
        # if circle_image is active this frame...
        if circle_image.status == STARTED:
            # update params
            pass
        
        # *diamond_image* updates
        
        # if diamond_image is starting this frame...
        if diamond_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            diamond_image.frameNStart = frameN  # exact frame index
            diamond_image.tStart = t  # local t and not account for scr refresh
            diamond_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(diamond_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'diamond_image.started')
            # update status
            diamond_image.status = STARTED
            diamond_image.setAutoDraw(True)
        
        # if diamond_image is active this frame...
        if diamond_image.status == STARTED:
            # update params
            pass
        
        # *Disgusting_self_lable* updates
        
        # if Disgusting_self_lable is starting this frame...
        if Disgusting_self_lable.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Disgusting_self_lable.frameNStart = frameN  # exact frame index
            Disgusting_self_lable.tStart = t  # local t and not account for scr refresh
            Disgusting_self_lable.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Disgusting_self_lable, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Disgusting_self_lable.started')
            # update status
            Disgusting_self_lable.status = STARTED
            Disgusting_self_lable.setAutoDraw(True)
        
        # if Disgusting_self_lable is active this frame...
        if Disgusting_self_lable.status == STARTED:
            # update params
            pass
        
        # *Desirable_self_lable* updates
        
        # if Desirable_self_lable is starting this frame...
        if Desirable_self_lable.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Desirable_self_lable.frameNStart = frameN  # exact frame index
            Desirable_self_lable.tStart = t  # local t and not account for scr refresh
            Desirable_self_lable.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Desirable_self_lable, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Desirable_self_lable.started')
            # update status
            Desirable_self_lable.status = STARTED
            Desirable_self_lable.setAutoDraw(True)
        
        # if Desirable_self_lable is active this frame...
        if Desirable_self_lable.status == STARTED:
            # update params
            pass
        
        # *key_resp_8* updates
        waitOnFlip = False
        
        # if key_resp_8 is starting this frame...
        if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_8.frameNStart = frameN  # exact frame index
            key_resp_8.tStart = t  # local t and not account for scr refresh
            key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_8.started')
            # update status
            key_resp_8.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_8.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_8.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_8_allKeys.extend(theseKeys)
            if len(_key_resp_8_allKeys):
                key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_11* updates
        
        # if text_11 is starting this frame...
        if text_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_11.frameNStart = frameN  # exact frame index
            text_11.tStart = t  # local t and not account for scr refresh
            text_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_11.started')
            # update status
            text_11.status = STARTED
            text_11.setAutoDraw(True)
        
        # if text_11 is active this frame...
        if text_11.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_intro_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_intro_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_intro_2" ---
    for thisComponent in self_intro_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_intro_2
    self_intro_2.tStop = globalClock.getTime(format='float')
    self_intro_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_intro_2.stopped', self_intro_2.tStop)
    # check responses
    if key_resp_8.keys in ['', [], None]:  # No response was made
        key_resp_8.keys = None
    thisExp.addData('key_resp_8.keys',key_resp_8.keys)
    if key_resp_8.keys != None:  # we had a response
        thisExp.addData('key_resp_8.rt', key_resp_8.rt)
        thisExp.addData('key_resp_8.duration', key_resp_8.duration)
    thisExp.nextEntry()
    # the Routine "self_intro_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "self_starting_3" ---
    # create an object to store info about Routine self_starting_3
    self_starting_3 = data.Routine(
        name='self_starting_3',
        components=[text_12, key_resp_9],
    )
    self_starting_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_9
    key_resp_9.keys = []
    key_resp_9.rt = []
    _key_resp_9_allKeys = []
    # store start times for self_starting_3
    self_starting_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    self_starting_3.tStart = globalClock.getTime(format='float')
    self_starting_3.status = STARTED
    thisExp.addData('self_starting_3.started', self_starting_3.tStart)
    self_starting_3.maxDuration = None
    # keep track of which components have finished
    self_starting_3Components = self_starting_3.components
    for thisComponent in self_starting_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "self_starting_3" ---
    self_starting_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_12* updates
        
        # if text_12 is starting this frame...
        if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_12.frameNStart = frameN  # exact frame index
            text_12.tStart = t  # local t and not account for scr refresh
            text_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_12.started')
            # update status
            text_12.status = STARTED
            text_12.setAutoDraw(True)
        
        # if text_12 is active this frame...
        if text_12.status == STARTED:
            # update params
            pass
        
        # *key_resp_9* updates
        waitOnFlip = False
        
        # if key_resp_9 is starting this frame...
        if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_9.frameNStart = frameN  # exact frame index
            key_resp_9.tStart = t  # local t and not account for scr refresh
            key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_9.started')
            # update status
            key_resp_9.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_9.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_9.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_9_allKeys.extend(theseKeys)
            if len(_key_resp_9_allKeys):
                key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            self_starting_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in self_starting_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "self_starting_3" ---
    for thisComponent in self_starting_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for self_starting_3
    self_starting_3.tStop = globalClock.getTime(format='float')
    self_starting_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('self_starting_3.stopped', self_starting_3.tStop)
    # check responses
    if key_resp_9.keys in ['', [], None]:  # No response was made
        key_resp_9.keys = None
    thisExp.addData('key_resp_9.keys',key_resp_9.keys)
    if key_resp_9.keys != None:  # we had a response
        thisExp.addData('key_resp_9.rt', key_resp_9.rt)
        thisExp.addData('key_resp_9.duration', key_resp_9.duration)
    thisExp.nextEntry()
    # the Routine "self_starting_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    p_trials_2 = data.TrialHandler2(
        name='p_trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('B_block2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(p_trials_2)  # add the loop to the experiment
    thisP_trial_2 = p_trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisP_trial_2.rgb)
    if thisP_trial_2 != None:
        for paramName in thisP_trial_2:
            globals()[paramName] = thisP_trial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisP_trial_2 in p_trials_2:
        currentLoop = p_trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisP_trial_2.rgb)
        if thisP_trial_2 != None:
            for paramName in thisP_trial_2:
                globals()[paramName] = thisP_trial_2[paramName]
        
        # --- Prepare to start Routine "practice2" ---
        # create an object to store info about Routine practice2
        practice2 = data.Routine(
            name='practice2',
            components=[p_fixation_2, p_fixation_4, p_text_8, key_resp_6_p, p_image_2],
        )
        practice2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        p_text_8.setText(label_2b)
        # create starting attributes for key_resp_6_p
        key_resp_6_p.keys = []
        key_resp_6_p.rt = []
        _key_resp_6_p_allKeys = []
        p_image_2.setImage(stimulus_2b)
        # store start times for practice2
        practice2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice2.tStart = globalClock.getTime(format='float')
        practice2.status = STARTED
        thisExp.addData('practice2.started', practice2.tStart)
        practice2.maxDuration = None
        # keep track of which components have finished
        practice2Components = practice2.components
        for thisComponent in practice2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice2" ---
        # if trial has changed, end Routine now
        if isinstance(p_trials_2, data.TrialHandler2) and thisP_trial_2.thisN != p_trials_2.thisTrial.thisN:
            continueRoutine = False
        practice2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *p_fixation_2* updates
            
            # if p_fixation_2 is starting this frame...
            if p_fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_2.frameNStart = frameN  # exact frame index
                p_fixation_2.tStart = t  # local t and not account for scr refresh
                p_fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_2.started')
                # update status
                p_fixation_2.status = STARTED
                p_fixation_2.setAutoDraw(True)
            
            # if p_fixation_2 is active this frame...
            if p_fixation_2.status == STARTED:
                # update params
                pass
            
            # if p_fixation_2 is stopping this frame...
            if p_fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fixation_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fixation_2.tStop = t  # not accounting for scr refresh
                    p_fixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fixation_2.stopped')
                    # update status
                    p_fixation_2.status = FINISHED
                    p_fixation_2.setAutoDraw(False)
            
            # *p_fixation_4* updates
            
            # if p_fixation_4 is starting this frame...
            if p_fixation_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_fixation_4.frameNStart = frameN  # exact frame index
                p_fixation_4.tStart = t  # local t and not account for scr refresh
                p_fixation_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_fixation_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_fixation_4.started')
                # update status
                p_fixation_4.status = STARTED
                p_fixation_4.setAutoDraw(True)
            
            # if p_fixation_4 is active this frame...
            if p_fixation_4.status == STARTED:
                # update params
                pass
            
            # if p_fixation_4 is stopping this frame...
            if p_fixation_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_fixation_4.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_fixation_4.tStop = t  # not accounting for scr refresh
                    p_fixation_4.tStopRefresh = tThisFlipGlobal  # on global time
                    p_fixation_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_fixation_4.stopped')
                    # update status
                    p_fixation_4.status = FINISHED
                    p_fixation_4.setAutoDraw(False)
            
            # *p_text_8* updates
            
            # if p_text_8 is starting this frame...
            if p_text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_text_8.frameNStart = frameN  # exact frame index
                p_text_8.tStart = t  # local t and not account for scr refresh
                p_text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_text_8.started')
                # update status
                p_text_8.status = STARTED
                p_text_8.setAutoDraw(True)
            
            # if p_text_8 is active this frame...
            if p_text_8.status == STARTED:
                # update params
                pass
            
            # if p_text_8 is stopping this frame...
            if p_text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_text_8.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_text_8.tStop = t  # not accounting for scr refresh
                    p_text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    p_text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_text_8.stopped')
                    # update status
                    p_text_8.status = FINISHED
                    p_text_8.setAutoDraw(False)
            
            # *key_resp_6_p* updates
            waitOnFlip = False
            
            # if key_resp_6_p is starting this frame...
            if key_resp_6_p.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6_p.frameNStart = frameN  # exact frame index
                key_resp_6_p.tStart = t  # local t and not account for scr refresh
                key_resp_6_p.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6_p, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_6_p.started')
                # update status
                key_resp_6_p.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6_p.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6_p.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_6_p is stopping this frame...
            if key_resp_6_p.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_6_p.tStartRefresh + 1.1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_6_p.tStop = t  # not accounting for scr refresh
                    key_resp_6_p.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_6_p.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_6_p.stopped')
                    # update status
                    key_resp_6_p.status = FINISHED
                    key_resp_6_p.status = FINISHED
            if key_resp_6_p.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6_p.getKeys(keyList=["m","z"], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_p_allKeys.extend(theseKeys)
                if len(_key_resp_6_p_allKeys):
                    key_resp_6_p.keys = _key_resp_6_p_allKeys[-1].name  # just the last key pressed
                    key_resp_6_p.rt = _key_resp_6_p_allKeys[-1].rt
                    key_resp_6_p.duration = _key_resp_6_p_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *p_image_2* updates
            
            # if p_image_2 is starting this frame...
            if p_image_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                p_image_2.frameNStart = frameN  # exact frame index
                p_image_2.tStart = t  # local t and not account for scr refresh
                p_image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_image_2.started')
                # update status
                p_image_2.status = STARTED
                p_image_2.setAutoDraw(True)
            
            # if p_image_2 is active this frame...
            if p_image_2.status == STARTED:
                # update params
                pass
            
            # if p_image_2 is stopping this frame...
            if p_image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_image_2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    p_image_2.tStop = t  # not accounting for scr refresh
                    p_image_2.tStopRefresh = tThisFlipGlobal  # on global time
                    p_image_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_image_2.stopped')
                    # update status
                    p_image_2.status = FINISHED
                    p_image_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice2" ---
        for thisComponent in practice2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice2
        practice2.tStop = globalClock.getTime(format='float')
        practice2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice2.stopped', practice2.tStop)
        # check responses
        if key_resp_6_p.keys in ['', [], None]:  # No response was made
            key_resp_6_p.keys = None
        p_trials_2.addData('key_resp_6_p.keys',key_resp_6_p.keys)
        if key_resp_6_p.keys != None:  # we had a response
            p_trials_2.addData('key_resp_6_p.rt', key_resp_6_p.rt)
            p_trials_2.addData('key_resp_6_p.duration', key_resp_6_p.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice2.maxDurationReached:
            routineTimer.addTime(-practice2.maxDuration)
        elif practice2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.600000)
        
        # --- Prepare to start Routine "feedback_2_p" ---
        # create an object to store info about Routine feedback_2_p
        feedback_2_p = data.Routine(
            name='feedback_2_p',
            components=[text_16],
        )
        feedback_2_p.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        # Check if they responded correctly, incorrectly, or missed
        is_match1 = int(is_match_2b)  # is_match_2b is now treated as a variable from the Excel file
        
        if not key_resp_6_p.keys:  # No response was made
            feedback_message_2 = 'Miss'
        elif key_resp_6_p.keys == 'm' and is_match1:  # Correct match response
            feedback_message_2 = 'Correct'
        elif key_resp_6_p.keys == 'z' and not is_match1:  # Correct not-match response
            feedback_message_2 = 'Correct'
        else:  # Any other response is incorrect
            feedback_message_2 = 'Incorrect'
           
        
        
        text_16.setText(feedback_message_2)
        # store start times for feedback_2_p
        feedback_2_p.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_2_p.tStart = globalClock.getTime(format='float')
        feedback_2_p.status = STARTED
        thisExp.addData('feedback_2_p.started', feedback_2_p.tStart)
        feedback_2_p.maxDuration = None
        # keep track of which components have finished
        feedback_2_pComponents = feedback_2_p.components
        for thisComponent in feedback_2_p.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_2_p" ---
        # if trial has changed, end Routine now
        if isinstance(p_trials_2, data.TrialHandler2) and thisP_trial_2.thisN != p_trials_2.thisTrial.thisN:
            continueRoutine = False
        feedback_2_p.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_16* updates
            
            # if text_16 is starting this frame...
            if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_16.frameNStart = frameN  # exact frame index
                text_16.tStart = t  # local t and not account for scr refresh
                text_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_16.started')
                # update status
                text_16.status = STARTED
                text_16.setAutoDraw(True)
            
            # if text_16 is active this frame...
            if text_16.status == STARTED:
                # update params
                pass
            
            # if text_16 is stopping this frame...
            if text_16.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_16.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_16.tStop = t  # not accounting for scr refresh
                    text_16.tStopRefresh = tThisFlipGlobal  # on global time
                    text_16.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_16.stopped')
                    # update status
                    text_16.status = FINISHED
                    text_16.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_2_p.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_2_p.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_2_p" ---
        for thisComponent in feedback_2_p.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_2_p
        feedback_2_p.tStop = globalClock.getTime(format='float')
        feedback_2_p.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_2_p.stopped', feedback_2_p.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback_2_p.maxDurationReached:
            routineTimer.addTime(-feedback_2_p.maxDuration)
        elif feedback_2_p.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'p_trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start2" ---
    # create an object to store info about Routine start2
    start2 = data.Routine(
        name='start2',
        components=[start_2, key_resp_11],
    )
    start2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_11
    key_resp_11.keys = []
    key_resp_11.rt = []
    _key_resp_11_allKeys = []
    # store start times for start2
    start2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start2.tStart = globalClock.getTime(format='float')
    start2.status = STARTED
    thisExp.addData('start2.started', start2.tStart)
    start2.maxDuration = None
    # keep track of which components have finished
    start2Components = start2.components
    for thisComponent in start2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start2" ---
    start2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_2* updates
        
        # if start_2 is starting this frame...
        if start_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_2.frameNStart = frameN  # exact frame index
            start_2.tStart = t  # local t and not account for scr refresh
            start_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_2.started')
            # update status
            start_2.status = STARTED
            start_2.setAutoDraw(True)
        
        # if start_2 is active this frame...
        if start_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_11* updates
        waitOnFlip = False
        
        # if key_resp_11 is starting this frame...
        if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_11.frameNStart = frameN  # exact frame index
            key_resp_11.tStart = t  # local t and not account for scr refresh
            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_11.started')
            # update status
            key_resp_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_11.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_11_allKeys.extend(theseKeys)
            if len(_key_resp_11_allKeys):
                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start2" ---
    for thisComponent in start2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start2
    start2.tStop = globalClock.getTime(format='float')
    start2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start2.stopped', start2.tStop)
    # check responses
    if key_resp_11.keys in ['', [], None]:  # No response was made
        key_resp_11.keys = None
    thisExp.addData('key_resp_11.keys',key_resp_11.keys)
    if key_resp_11.keys != None:  # we had a response
        thisExp.addData('key_resp_11.rt', key_resp_11.rt)
        thisExp.addData('key_resp_11.duration', key_resp_11.duration)
    thisExp.nextEntry()
    # the Routine "start2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('B_block2.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "trial_2" ---
        # create an object to store info about Routine trial_2
        trial_2 = data.Routine(
            name='trial_2',
            components=[fixation_2, fixation_4, text_8, key_resp_6, image_2],
        )
        trial_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_8.setText(label_2b)
        # create starting attributes for key_resp_6
        key_resp_6.keys = []
        key_resp_6.rt = []
        _key_resp_6_allKeys = []
        image_2.setImage(stimulus_2b)
        # store start times for trial_2
        trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_2.tStart = globalClock.getTime(format='float')
        trial_2.status = STARTED
        thisExp.addData('trial_2.started', trial_2.tStart)
        trial_2.maxDuration = None
        # keep track of which components have finished
        trial_2Components = trial_2.components
        for thisComponent in trial_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        trial_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.6:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2.started')
                # update status
                fixation_2.status = STARTED
                fixation_2.setAutoDraw(True)
            
            # if fixation_2 is active this frame...
            if fixation_2.status == STARTED:
                # update params
                pass
            
            # if fixation_2 is stopping this frame...
            if fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_2.stopped')
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # *fixation_4* updates
            
            # if fixation_4 is starting this frame...
            if fixation_4.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                fixation_4.frameNStart = frameN  # exact frame index
                fixation_4.tStart = t  # local t and not account for scr refresh
                fixation_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_4.started')
                # update status
                fixation_4.status = STARTED
                fixation_4.setAutoDraw(True)
            
            # if fixation_4 is active this frame...
            if fixation_4.status == STARTED:
                # update params
                pass
            
            # if fixation_4 is stopping this frame...
            if fixation_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_4.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_4.tStop = t  # not accounting for scr refresh
                    fixation_4.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_4.stopped')
                    # update status
                    fixation_4.status = FINISHED
                    fixation_4.setAutoDraw(False)
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            
            # *key_resp_6* updates
            waitOnFlip = False
            
            # if key_resp_6 is starting this frame...
            if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6.frameNStart = frameN  # exact frame index
                key_resp_6.tStart = t  # local t and not account for scr refresh
                key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_6.started')
                # update status
                key_resp_6.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_6 is stopping this frame...
            if key_resp_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_6.tStartRefresh + 1.1-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_6.tStop = t  # not accounting for scr refresh
                    key_resp_6.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_6.stopped')
                    # update status
                    key_resp_6.status = FINISHED
                    key_resp_6.status = FINISHED
            if key_resp_6.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6.getKeys(keyList=["m","z"], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_allKeys.extend(theseKeys)
                if len(_key_resp_6_allKeys):
                    key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                    key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                    key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *image_2* updates
            
            # if image_2 is starting this frame...
            if image_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.started')
                # update status
                image_2.status = STARTED
                image_2.setAutoDraw(True)
            
            # if image_2 is active this frame...
            if image_2.status == STARTED:
                # update params
                pass
            
            # if image_2 is stopping this frame...
            if image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_2.tStop = t  # not accounting for scr refresh
                    image_2.tStopRefresh = tThisFlipGlobal  # on global time
                    image_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.stopped')
                    # update status
                    image_2.status = FINISHED
                    image_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_2" ---
        for thisComponent in trial_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_2
        trial_2.tStop = globalClock.getTime(format='float')
        trial_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_2.stopped', trial_2.tStop)
        # check responses
        if key_resp_6.keys in ['', [], None]:  # No response was made
            key_resp_6.keys = None
        trials_2.addData('key_resp_6.keys',key_resp_6.keys)
        if key_resp_6.keys != None:  # we had a response
            trials_2.addData('key_resp_6.rt', key_resp_6.rt)
            trials_2.addData('key_resp_6.duration', key_resp_6.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_2.maxDurationReached:
            routineTimer.addTime(-trial_2.maxDuration)
        elif trial_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.600000)
        
        # --- Prepare to start Routine "feedback_2" ---
        # create an object to store info about Routine feedback_2
        feedback_2 = data.Routine(
            name='feedback_2',
            components=[text_9],
        )
        feedback_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        # Check if they responded correctly, incorrectly, or missed
        is_match1 = int(is_match_2b)  # is_match_2b is now treated as a variable from the Excel file
        
        if not key_resp_6.keys:  # No response was made
            feedback_message_2 = 'Miss'
        elif key_resp_6.keys == 'm' and is_match1:  # Correct match response
            feedback_message_2 = 'Correct'
        elif key_resp_6.keys == 'z' and not is_match1:  # Correct not-match response
            feedback_message_2 = 'Correct'
        else:  # Any other response is incorrect
            feedback_message_2 = 'Incorrect'
           
        
        
        text_9.setText(feedback_message_2)
        # store start times for feedback_2
        feedback_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_2.tStart = globalClock.getTime(format='float')
        feedback_2.status = STARTED
        thisExp.addData('feedback_2.started', feedback_2.tStart)
        feedback_2.maxDuration = None
        # keep track of which components have finished
        feedback_2Components = feedback_2.components
        for thisComponent in feedback_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        feedback_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # if text_9 is stopping this frame...
            if text_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_9.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_9.tStop = t  # not accounting for scr refresh
                    text_9.tStopRefresh = tThisFlipGlobal  # on global time
                    text_9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_9.stopped')
                    # update status
                    text_9.status = FINISHED
                    text_9.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_2" ---
        for thisComponent in feedback_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_2
        feedback_2.tStop = globalClock.getTime(format='float')
        feedback_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_2.stopped', feedback_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback_2.maxDurationReached:
            routineTimer.addTime(-feedback_2.maxDuration)
        elif feedback_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Thank_you" ---
    # create an object to store info about Routine Thank_you
    Thank_you = data.Routine(
        name='Thank_you',
        components=[text_7],
    )
    Thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Thank_you
    Thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Thank_you.tStart = globalClock.getTime(format='float')
    Thank_you.status = STARTED
    thisExp.addData('Thank_you.started', Thank_you.tStart)
    Thank_you.maxDuration = None
    # keep track of which components have finished
    Thank_youComponents = Thank_you.components
    for thisComponent in Thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thank_you" ---
    Thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # if text_7 is stopping this frame...
        if text_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_7.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_7.tStop = t  # not accounting for scr refresh
                text_7.tStopRefresh = tThisFlipGlobal  # on global time
                text_7.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.stopped')
                # update status
                text_7.status = FINISHED
                text_7.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thank_you" ---
    for thisComponent in Thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Thank_you
    Thank_you.tStop = globalClock.getTime(format='float')
    Thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Thank_you.stopped', Thank_you.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Thank_you.maxDurationReached:
        routineTimer.addTime(-Thank_you.maxDuration)
    elif Thank_you.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
