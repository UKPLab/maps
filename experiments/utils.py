def generate_statistics_and_report(total, correct, incorrect, pattern, model_name, output_path):
    msg = f"####\n Model name: {model_name}\t "
    msg += f"Pattern: {pattern}\t"
    total = min(total, correct + incorrect)
    msg += f"correct: {correct}, incorrect: {incorrect}, total: {total}\t"
    msg += f"correct %: {correct / total}\n"
    print (msg)
    with open(output_path, 'a') as f:
        f.write(msg)
    return correct / total


def generate_group_statistics_and_report(grouping_acc, model_name, output_path):
    msg = f"####\n Model name: {model_name}\t "
    acc = grouping_acc[1][0] / max(1, sum(grouping_acc[1]))
    msg += f"Group1: {grouping_acc[1]}, {acc}\t"
    acc = grouping_acc[2][0] / max(1, sum(grouping_acc[2]))
    msg += f"Group2: {grouping_acc[2]}, {acc}\t"
    a = sum(grouping_acc[2]) / (sum(grouping_acc[1]) + sum(grouping_acc[2]))
    msg += f"Mem frac: {a}"
    print (msg)
    with open(output_path, 'a') as f:
        f.write(msg)


def generate_all(all_acc, model_name, lang, output_path, memorized=None):
    msg = f"####\n Model name: {model_name}\tLanguage: {lang}\t"
    msg += f"Average acc %: {sum(all_acc) / len(all_acc)}\t"
    if memorized:
        msg += f"Mem: {memorized}\t"
    msg += f"Max acc %: {max(all_acc)}\n####\n"
    print (msg)
    with open(output_path, 'a') as f:
        f.write(msg)
