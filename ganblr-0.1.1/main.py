import time

from ganblr import get_demo_data
from ganblr.models import GANBLR
from ucimlrepo import fetch_ucirepo

# # this is a discrete version of adult since GANBLR requires discrete data.
# df = get_demo_data('adult')

#Todo list:
# Done 1. Feed the UCI Database into the GANBLR
# 2. Leverage the NOTEARS into the GANBLR
# 2-1. The Integration of Notears: Create a new one using fit function as the parameter
# 2-2. The Modification of the fit: The parameter learning of the NOTEARS DAG
#   Two kinds: Calculate from the Dataset/Learned
# 3. Sampling seems specially designed for KDB
# 4. Integrate KDB


# fetch dataset
def get_uci_data(name="adult"):
    if name == "adult":
        dataset = fetch_ucirepo(id=2)
    elif name == "intrusion":
        dataset = fetch_ucirepo(id=942) #Maybe the wrong one?
    elif name == "pokerhand":
        dataset = fetch_ucirepo(id=158)
    elif name == "shuttle":
        dataset = fetch_ucirepo(id=148)
    elif name == "connect":
        dataset = fetch_ucirepo(id=151)
    elif name == "chess":
        dataset = fetch_ucirepo(id=22)
    elif name == "letter": #letter recognition
        dataset = fetch_ucirepo(id=59)
    elif name == "magic":
        dataset = fetch_ucirepo(id=159)
    elif name == "nursery":
        dataset = fetch_ucirepo(id=76)
    elif name == "satellite":
        dataset = fetch_ucirepo(id=146)
    elif name == "car":
        dataset = fetch_ucirepo(id=19)
    else:
        raise Exception("Please Check Your Dataset Name")
    df = dataset.data.original.dropna(axis=0)
    return df

def test_ganblr(name="adult"):
    df = get_uci_data(name=name)
    x, y = df.values[:,:-1], df.values[:,-1]
    # x: Dataset to fit the model.
    # y: Label of the dataset.

    model = GANBLR()
    start_time = time.time()
    model.fit(x, y, epochs = 100)
    end_time = time.time()


    lr_result = model.evaluate(x, y, model='lr')
    mlp_result = model.evaluate(x, y, model='mlp')
    rf_result = model.evaluate(x, y, model='rf')

    results = {
        "Logistic Regression": lr_result,
        "MLP": mlp_result,
        "Random Forest": rf_result
    }
    print("Dataset:",name)
    print("Training time:", (end_time-start_time), "seconds")
    for model_name, result in results.items():
        print(f"{model_name}: {result}")

    file_path = "./running_result.txt"
    with open(file_path,"a") as f:
        f.write("Dataset:",name)
        for model_name, result in results.items():
            f.write(f"{model_name}: {result}")

    return

if __name__ == '__main__':
    available_datasets = ["adult","intrusion","pokerhand","shuttle","connect","chess","letter","magic","nursery","satellite","car"]
    print("Testing the following datasets:",available_datasets)
    for dataset_name in available_datasets:
        print("Start test: ", dataset_name)
        test_ganblr(name=dataset_name)
