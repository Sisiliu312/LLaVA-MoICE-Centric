#!/usr/bin/env python3
import json
import argparse

def load_predictions(pred_file):
    """
    预测文件是 jsonl，每行一个 json。
    返回字典：{ question_id : predicted_answer }
    """
    preds = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            qid = j["question_id"]
            # 统一大小写去空格
            pred = j["text"].strip().upper()
            preds[qid] = pred
    return preds

def load_groundtruth(gt_file):
    """
    groundtruth 是一个 json list。
    返回字典：{ id : correct_answer }
    """
    gts = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        qid = item["id"]
        # gpt 的答案在 conversations 里 from == "gpt"
        conv = item["conversations"]
        gt_ans = None
        for c in conv:
            if c["from"].lower() == "gpt":
                gt_ans = c["value"].strip().upper()
                break
        if gt_ans is None:
            raise ValueError(f"No groundtruth answer found for id: {qid}")
        gts[qid] = gt_ans
    return gts

def eval_accuracy(preds, gts):
    """
    计算准确率
    """
    total = 0
    correct = 0
    for qid, gt in gts.items():
        if qid not in preds:
            print(f"Warning: {qid} not found in predictions")
            continue
        total += 1
        if preds[qid] == gt:
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return acc, correct, total

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of llava predictions")
    parser.add_argument("--result-file", required=True, help="模型预测 jsonl 文件路径")
    parser.add_argument("--annotation-file", required=True, help="groundtruth json 文件路径")
    args = parser.parse_args()

    preds = load_predictions(args.result_file)
    gts = load_groundtruth(args.annotation_file)
    acc, correct, total = eval_accuracy(preds, gts)

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()