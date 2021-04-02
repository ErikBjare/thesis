Experiment protocol
===================

Protocol for running EEG experiments with subject.


# Setup

 - EEG device
   - Which? 
     - Muse S?
     - Cyton + Ultracortex?

 - ActivityWatch
   - `aw-watcher-web` to collect browser data
   - `aw-watcher-input` to collect detailed input data
     - Set the poll time to something low (0.2s?)


# Protocol

## Before

 - Inform about collected data and anonymization process (we won't store names or other personally identifiable information)
 - Ask for informed consent
 - Take note of: 
   - gender?
   - device?
   - age?
   - software dev experience, specifically C/C++
 - Ask subject how they are feeling
   - Any standardized way?
   - States to ask about: tired, calm, focused, happy, sad
 - Ask subject if they've had caffeine today

## During

 - (optional, if Muse) optionally have a dummy meditation session using the Muse app
 - Put on the EEG device
 - Start eegwatch
   - Set `--duration` to a high value to prevent recording restarting
   - Set `--subject` to the subject ID
 - Do a signal check
   - Use `muselsl view -b Qt5Agg`
   - Use `eegwatch check`
 - Run codeprose task
 - (optional) Run arithmetic task
 - (optional) Other tasks?
   - 10min Twitter or 10min YouTube?
   - 10min of working through email? (include input watcher)
   - 10min of programming? (include input watcher)
   - Have them read Wikipedia articles on different subjects (lets say something technical vs historical)
     - Has this been done before?
     - Should probably find a set of ~5 articles/sections in each category

## After (debrief)

 - Ask subject how they are feeling again
   - If they answer the same, we'll assume they've been feeling that way for the duration of the test
 - Ask briefly about their experience
   - Comfort
   - Difficulty of tasks
 - ...
 - Collect data
   - EEG signals (duh)
   -
