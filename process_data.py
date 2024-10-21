# Compute Stats
import json

# TODO
def process(segfile = "./data/segmental_information.json"):

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

    print("Total Segments:", len(list(seg.values())))
    print("Total HQ Segments:", len([v for v in seg.values() if v['mi_quality']=="high"]))
    print("Total LQ Segments:", len([v for v in seg.values() if v['mi_quality']=="low"]))

    print("Total Sessions:", len(list(sessions.keys())))
    print("Total HQ Sessions:", len([k for k,v in sessions.items() if v[0]["mi_quality"]=="high"]))
    print("Total LQ Sessions:", len([k for k,v in sessions.items() if v[0]["mi_quality"]=="low"]))


    dic = {"train": [], "dev": [], "test": []}


    return dic