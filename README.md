# Speech MI - DJ 2024 Fall Project

## TODOs
- finish the code with all the fixins (validation, test, and saving and such) [first pass]
- do test runs and make sure all is functional [v] 241023 afternoon
- first implement the testing functionality:
    - collect not only predicted, target but interlocutor and other important metadata
- do the prelimnary tests
    - classification []
    - forecasting []
- TOP Priority: the generations are gibberish mostly....
 - try adding a special token to the end of the input (process_data.py)
 - MAYBE: in 'embed_audio_and_concatenate' function, the padding is currently
   -  text /  speech
   - left paddings, tokens, tokens,  / tokens, tokens, tokens ... right paddings
        - try rearranging so that the padding is strictly left padding
    ==> but this doesn't explain why the text version isn't also working
    (ob1): when no peft weight is used, it at least generates some continuation
        - (no training): 
            [("therapist: Okay, so I'm just gonna ask you a couple of more background questions.\nclient: Okay.\ntherapist: Do you use any tobacco products, or smoke cigarettes?\n[SEP]", "\nclient: No, I don't smoke.\ntherapist: And have you ever been diagnosed with any mental health conditions?\nclient: No, I haven't.\ntherapist: Okay. And how would you describe your relationship with your mother?\nclient: I would say that I have a good relationship with her. We get along pretty well.\ntherapist: And how would you describe your relationship with your father?\nclient: I would say that I have a strained relationship with him. We don't really talk to each other.\ntherapist: And how would you describe your relationship with your stepmother?\nclient: I would say that I have a positive relationship with her. We get along pretty well.\ntherapist: And how would you describe your relationship with your brother?\nclient: I would say that I have a good relationship with him. We get along pretty well.\ntherapist: And how would you describe your relationship with your sister?\nclient: I would say that I have a good relationship with her. We get along pretty well.\ntherapist: And how would you describe your relationship with your nephew?\nclient: I would say", 'question', 'therapist')]

        - with some training: 
            [("therapist: Okay, so I'm just gonna ask you a couple of more background questions.\nclient: Okay.\ntherapist: Do you use any tobacco products, or smoke cigarettes?\n[SEP]", '1. Field of the Invention\nThe present invention relates to a system and method for a wireless communication system, and more particularly to a system and method for a wireless communication system that uses a plurality of wireless communication devices.\n2. Description of the Related Art\nWireless communication systems have become increasingly prevalent in modern society. These systems allow for communication between devices without the need for physical connections, such as cables. One common wireless communication system is a wireless local area network (WLAN), which is used to connect devices within a localized area, such as a home or office.\nWLANs typically use a centralized access point (AP) that connects to a wired network, such as the internet or a local area network (LAN). The AP broadcasts wireless signals that can be received by wireless communication devices, such as wireless network adapters (WNADs) or wireless network bridges (WNBs), within the range of the AP.\nWireless communication devices typically communicate with the AP using a protocol, such as the IEEE 802.11 standard. The AP acts as a central hub for communication between the wireless communication devices and', 'question', 'therapist')]
    ==> this is fucking easy task and loss isn't going down to 0.0....
        must be the loss computation that's fucked up
        or maybe it takes time? but the the 
    - [image](assets/curve241024.png)

    - ALso if can't fix then just do classification
    - i think the loss computation in audio_llama.py is wrong
        - [IMPORTANT]: I think I figured out why
                       in the convention of the audio_llama code, they were 
                       providing
## Model and Framework
- LLM + Speech Adapter
- We can use the srag codebase...
    - 7b
    - Hubert Encoder Adapter
- Learning Objective (https://arxiv.org/pdf/2406.05968)
    - Next Token Prediction (NTP)
    - token logit distillation (LD)
- How to train them
    - Unified model with input formatting
    - Separet model for each tasks

### Experiment 1: Counselor Response Prediction

### Experiment 2: Client Modeling



## RQ & Hypothesis **
- Which framework: Speech-based MI and Conversation Modeling that consumses client utterance in **speech form, not transcribed form**,
- is different how: it's better for MI perfornace on tasks:
    - counselor response prediction
    - client understanding
    - etc
- from which previous baseline: text only
- because of which aspect
    - paralinguistic information and stuff
    - but how show


## Which Specific Tasks?

Two tasks I came up with
- Counselor Response Prediction (perplexity? text similarity?)
- Counseling Quality Estimation
- Prediction of Client Attitude and stuff
    - Client talk type


## Man Repo
https://www.overleaf.com/project/66d8ad52258808d4782fe8df