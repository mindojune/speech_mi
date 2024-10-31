# Speech MI - DJ 2024 Fall Project

## TODOs


- 241031 cont from dojmin@gl-login3 
    - STATUS: PADDING BUG THAT LEAD TO DIFF BSZ WITH DIFF RESULT IS .... FIXED MOOFOOGASSSSSS!!!!
    - MAIN Lesson: I'm definitely messing something up with the padding
                    MOST LIKELY in embed_audio_and_concatenate
                    and MOST LIKELY during the concatenation part (of text and audio)
    - new_info: yes it's something related to bsz
        - IS it in the left padding alignment? L326 "if True..."
            ==> BSZ 1
                    accuracy                           0.71       663
                    macro avg       0.52      0.47      0.47       663
                    weighted avg       0.71      0.71      0.68       663
                BSZ 4
                    accuracy                           0.62       663
                    macro avg       0.25      0.23      0.23       663
                    weighted avg       0.62      0.62      0.60       663
            ==> fixed and 12 samples
                BSZ = 1
                    accuracy                           0.83        12
                    macro avg       0.46      0.46      0.45        12
                    weighted avg       0.85      0.83      0.83        12

                    
                BSZ = 2
                    accuracy                           0.83        12
                    macro avg       0.46      0.46      0.45        12
                    weighted avg       0.85      0.83      0.83        12

                BSZ = 4
                    accuracy                           0.83        12
                    macro avg       0.46      0.46      0.45        12
                    weighted avg       0.85      0.83      0.83        12
            ==> fixed and full samples
                BSZ = 1 
                    accuracy                           0.71       663
                    macro avg       0.52      0.47      0.47       663
                    weighted avg       0.71      0.71      0.68       663
                BSZ = 2
                accuracy                           0.71       663
                macro avg       0.52      0.47      0.47       663
                weighted avg       0.71      0.71      0.68       663

                BSZ = 4




        - EVIDENCE: same model, different test_batch_size leads to WILDY different results
        BSZ = 4
                accuracy                           0.65       663
                macro avg       0.61      0.64      0.61       663
                weighted avg       0.69      0.65      0.66       663


        BSZ = 2
                accuracy                           0.67       663
                macro avg       0.54      0.57      0.54       663
                weighted avg       0.71      0.67      0.68       663

                
        BSZ = 1
                accuracy                           0.73       663
                macro avg       0.68      0.70      0.68       663
                weighted avg       0.76      0.73      0.73       663

        different model 1e-4 without eos
        BSZ = 2
                accuracy                           0.67        12
                macro avg       0.31      0.25      0.26        12
                weighted avg       0.85      0.67      0.71        12

        BSZ = 1
                accuracy                           0.83        12
                macro avg       0.46      0.46      0.45        12
                weighted avg       0.85      0.83      0.83        12

        FINAL test, if this phenom persists in text mode.... 
            if yes then problem could be in shared code
            if not the problem is in speech exclusive part
        text, BSZ = 4

            accuracy                           0.69       663
            macro avg       0.72      0.58      0.58       663
            weighted avg       0.74      0.69      0.67       663


        text, BSZ = 2

            accuracy                           0.69       663
            macro avg       0.72      0.58      0.58       663
            weighted avg       0.74      0.69      0.67       663


        text, BSZ = 1
            accuracy                           0.69       663
            macro avg       0.72      0.58      0.58       663
            weighted avg       0.74      0.69      0.66       663

         ==> so yeah it's most def somethign wrong with  BSZ = masking, padding of the AUDIO part (subhanallah)
    - yes eos
        - ?
    - still the best performance is on frozen encoder + 1e-4 lora
    - args.max_length = 512 [todo]
        - see if the audio is too long and thus cutting information
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 355])
        input_features: torch.Size([1, 73, 3072])
        2%|██▏                                                                                                                                                  | 10/663 [00:03<03:34,  3.04batch/s]******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 401])
        input_features: torch.Size([1, 98, 3072])
        2%|██▍                                                                                                                                                  | 11/663 [00:03<03:35,  3.02batch/s]******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 406])
        input_features: torch.Size([1, 23, 3072])
        2%|██▋                                                                                                                                                  | 12/663 [00:04<03:34,  3.04batch/s]
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 490])
        input_features: torch.Size([1, 186, 3072])
        2%|██▉                                                                                                                                                  | 13/663 [00:04<03:42,  2.92batch/s]
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 512])
        input_features: torch.Size([1, 336, 3072])
        2%|███▏                                                                                                                                                 | 14/663 [00:04<03:51,  2.80batch/s]
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 512])
        input_features: torch.Size([1, 36, 3072])
        2%|███▎                                                                                                                                                 | 15/663 [00:05<03:48,  2.84batch/s]
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 512])
        input_features: torch.Size([1, 123, 3072])
        2%|███▌                                                                                                                                                 | 16/663 [00:05<03:46,  2.85batch/s]
        ******************************
        inside embed_audio_and_concatenate
        input_ids: torch.Size([1, 512])
        input_features: torch.Size([1, 36, 3072])


    - i am skipping resampling now since i already converted
        - but we can turn it back on to see it help with loss [todo]

- 241029-30
    - no eos: doesn't seem to be helping a lot

    - left padding aligned: does seem to be helping!!!! [done, but helpfulness limited]
        - it does help but at epoch5 the performance gap is still there
        - it only marginally helps

    - so really need to figure out what's going on [todo]
        - trying with train bsz = 1 to make sure the bsz isn't the problem [better model, but not full catchup]
            let's wait and see...
        - try with freeze_encoder, learning_rate 1e-4 [doesn't seem to help alot]
        - don't use lora, just train encoder... [doesn't seem to help alot]
            - if not good enough, train the lora with the trained encoder [todo]
        - other possibilities
            - something wrong with the audio processing
                - different sr or something
                - timestamp is not good
            - 3b not nuff go 7b?
        - minor change test: after speech, in front of label add a special token [todo]
            - probably not essential but could help with the training
                (i did with the eos embedding) no help
                but add [cls]

        
- 241025
    - speech loss converges at a high level compared to text loss
        possibilities
            - some bug in the code for speech processing
            - because we need to train more (speech adapter + lora) so it's kinda expected
                - freeze adapter (so that it doesn't have to be trained, but important that we use the pretrained one) [todo]
                - train longer [doing]

    - The performance of the speech model does not seem too competitive...
        This could be because we're jointly training the speech adapter and the lora model
        Thus first try to train the speech adapter then move on?
        but note I'm using a pretrained speech adapter..
            actually not as of now (241024),
            so make the change TODO [done!]
            ==> this didn't seem to help much
    
    - make the training go faster? [done!]
        ==> see if dummy data goes fasta (how much)
            - with real data 3s per batch (bsz = 2)
            - with dummy data 
        
- before 241025
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
                        so i added this part in embed and concatenate... make sure this is working as intended
                        so far loss curve is very diff, looking good

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
