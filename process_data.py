# Compute Stats
import json
import random
import torch
from functools import partial

def custom_datacollate(batch, task="classification"):
    """
    Collate function to process audio and other necessary data.
    TODO: audio processing and feeding ...
          Is it to be done here? ==> Do it in the training code with a separate function that takes in the mode ("text", "speech")
    """
    context = [item["context"] for item in batch]
    target = [item["target"] for item in batch]

    inputs = []
    labels = []
    audio_cue_infos = []

    for c, t in zip(context, target):
        # Interleave utterance_text from context
        interleaved_text = "\n".join([f"{utterance['interlocutor']}: {utterance['utterance_text']}" for utterance in c])

        # add a special token to separate the context from the target
        interleaved_text += f"\n[SEP]\n"  # change introduced on 241024 
        
        if task == "response_generation":
            interleaved_text += t["interlocutor"] + ": "
            label = t["utterance_text"]
        else:
            # Determine the label based on the interlocutor
            if t["interlocutor"] == "therapist":
                #labels.append(t["main_therapist_behaviour"])
                label = t["main_therapist_behaviour"]
            else:
                # labels.append(t["client_talk_type"])
                label = t["client_talk_type"]
        label += "\n[CLS]"
        # interleaved_text += " " + label # don't do this here 
        
        inputs.append(interleaved_text)
        labels.append(label)

        # Add the audio path, begin, and end of the last item in the context
        last_item = c[-1]
        audio_cue_infos.append({
            "audio_path": last_item["audio_path"],
            "begin": last_item["begin"],
            "end": last_item["end"],
            "interlocutor": t["interlocutor"]
        })

    return {"inputs": inputs, "labels": labels, "audio_infos": audio_cue_infos}

def process(args, split=[0.8,0.05,0.15], print_examples=False):
    # args = { }
    if args.dataset == "annomi":
        # segfile = "data/segmental_information.json"
        segfile = "data/converted_segmental_information.json"

    with open(segfile, "r") as fh:
        seg = json.load(fh)


    sessions = {}
    for seg_id, seg_info in seg.items():
        tid = seg_info["transcript_id"]
        if tid in sessions:
            sessions[tid] += [seg_info]
        else:
            sessions[tid] = [ seg_info]

    # original from AnnoMI paper https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9746035&tag=1
    #               HQ         LQ
    #Conversations 110 (82.7%) 23 (17.3%)
    #Utterances 8839 (91.1%) 860 (8.9%) 
    if False:
        print("Total Segments:", len(list(seg.values())))
        print("Total HQ Segments:", len([v for v in seg.values() if v['mi_quality']=="high"]))
        print("Total LQ Segments:", len([v for v in seg.values() if v['mi_quality']=="low"]))

        print("Total Sessions:", len(list(sessions.keys())))
        print("Total HQ Sessions:", len([k for k,v in sessions.items() if v[0]["mi_quality"]=="high"]))
        print("Total LQ Sessions:", len([k for k,v in sessions.items() if v[0]["mi_quality"]=="low"]))
    # divide sessions into train, dev, test using split
    # split = [0.8, 0.05, 0.15]
    #random.seed(42)  # For reproducibility

    session_keys = list(sessions.keys())
    #random.shuffle(session_keys)

    train_end = int(len(session_keys) * split[0])
    dev_end = train_end + int(len(session_keys) * split[1])

    train_sessions = session_keys[:train_end]
    dev_sessions = session_keys[train_end:dev_end]
    test_sessions = session_keys[dev_end:]

    def create_context_target_pairs(session):
        pairs = []
        for i in range(1, len(session)):

            if args.task == "classification":
                context = session[:i+1]
                target = session[i]
                # pairs.append({"context": context, "target": target})
            elif args.task == "forecasting":
                context = session[:i]
                target = session[i]
            elif args.task == "response_generation":
                # raise NotImplementedError
                context = session[:i]
                target = session[i]
                # pairs.append({"context": context, "target": target})
            # if begin == end for last item of context, skip or also skip when either of them is None
            if context[-1]["begin"] == context[-1]["end"] or context[-1]["begin"] is None or context[-1]["end"] is None:
                continue
            if context[-1]["begin"] >= context[-1]["end"]:
                continue
            # consider skipping when the time difference is small
            pairs.append({"context": context, "target": target})
        return pairs

    train_data = []
    dev_data = []
    test_data = []

    for tid in train_sessions:
        train_data.extend(create_context_target_pairs(sessions[tid]))

    for tid in dev_sessions:
        dev_data.extend(create_context_target_pairs(sessions[tid]))

    for tid in test_sessions:
        test_data.extend(create_context_target_pairs(sessions[tid]))

    if hasattr(args, 'data_length') and args.data_length is not None:
        train_len, dev_len, test_len = args.data_length
        # Adjust lengths if specified as -1
        train_len = train_len if train_len != -1 else len(train_data)
        dev_len = dev_len if dev_len != -1 else len(dev_data)
        test_len = test_len if test_len != -1 else len(test_data)

        train_data = train_data[:train_len]
        dev_data = dev_data[:dev_len]
        test_data = test_data[:test_len]

    custom_datacollate_task = partial(custom_datacollate, task=args.task)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_datacollate_task)
    devloader = torch.utils.data.DataLoader(dev_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_datacollate_task)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_datacollate_task)

    # TEST SAMPLING FROM DEVLOADER
    if print_examples:
        for i, batch in enumerate(devloader):
            
            print(i, batch)
            if i > 5:
                break

    # TODO: split data into train, dev, test and make them into dataloaders
    dic = { "train": trainloader, "dev": devloader, "test": testloader }
    return dic

if __name__ == "__main__":
    # make a namespace 
    args = { "dataset": "annomi", "task": "classification", "mode": ["train"], "run_name": "run1", "seed": 42, "batch_size": 1 }
    args = { "dataset": "annomi", "task": "forecasting", "mode": ["train"], "run_name": "run1", "seed": 42, "batch_size": 1 }
    
    from types import SimpleNamespace   
    args = SimpleNamespace(**args)
    process(args, print_examples=True)
