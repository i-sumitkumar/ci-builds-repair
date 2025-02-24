import json
import os
import time
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from benhmark_functions import get_results, process_datapoint, fix_apply_diff
from benchmark_utils import read_jsonl, save_jsonl

load_dotenv()


# Helper functions to filter files and by ID
def filter_files(directory, files):
    print(f"Filtering files in directory {directory}.")
    return [file for file in files if file != "meta_info.json"]


def filter_by_id(example, ids):
    print(f"Filtering example with id {example['id']}")
    return example['id'] in ids


class CIFixBenchmark:
    def __init__(self, model_name, config_path):
        print(f"Initializing CIFixBenchmark with model: {model_name} and config: {config_path}")
        benchmark_owner = "LCA-CI-builds-repair"
        self.dataset_id = "JetBrains-Research/lca-ci-builds-repair"

        # Load configuration
        self.config = OmegaConf.load(config_path)
        if not "test_username" in self.config:
            self.config.test_username = self.config.username_gh
        language = self.config.language
        self.credentials = {
            "username": self.config.username_gh,
            "token": os.environ.get("TOKEN_GH"),
            "model": model_name,
        }

        # Debugging print for credentials
        print(f"Configuration loaded: {self.config}")
        print(f"GitHub Token: {os.environ.get('TOKEN_GH')}")  # Ensure token is loaded

        # Create necessary directories
        os.makedirs(self.config.out_folder, exist_ok=True)
        os.makedirs(self.config.repos_folder, exist_ok=True)
        OmegaConf.update(
            self.config, "benchmark_owner", benchmark_owner, force_add=True
        )
        if hasattr(self.config, "data_cache_dir"):
            self.cache_dir = self.config.data_cache_dir
        else:
            self.cache_dir = None
        self.model_name = model_name

    def get_dataset(
            self, num_dp=None, force_download=False, dataset_folder=None
    ):
        print(f"Loading dataset from {dataset_folder if dataset_folder else self.dataset_id}")
        if dataset_folder is not None:
            self.dataset = load_dataset(path=dataset_folder)["train"]
            # TODO needs refactoring
            if num_dp is not None:
                self.dataset = self.dataset.select(range(num_dp))
            print(f"Dataset loaded from local folder, {len(self.dataset)} datapoints.")
            return self.dataset
        if force_download:
            download_mode = "force_redownload"
        else:
            download_mode = None
        self.dataset = load_dataset(
            self.dataset_id,
            cache_dir=self.cache_dir,
            download_mode=download_mode,
            split="test"
        )
        if num_dp is not None:
            self.dataset = self.dataset.select(range(num_dp))
        print(f"Dataset loaded from Hugging Face with {len(self.dataset)} datapoints.")
        return self.dataset


    def run_dataset(self, fix_repo_function, test_dataset=None):
        if test_dataset is None:
            test_dataset = self.dataset
        self.jobs_ids = []
        jobs_ids_file_path = os.path.join(
            self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
        )
        print(f"Running dataset with {len(test_dataset)} datapoints... Writing to {jobs_ids_file_path}")
        with open(jobs_ids_file_path, "w") as writer:
            for datapoint in tqdm(test_dataset):
                print(f"Processing datapoint: {datapoint['id']}")
                job_identificator = process_datapoint(
                    datapoint, fix_repo_function, self.config, self.credentials
                )
                self.jobs_ids.append(job_identificator)
                json.dump(job_identificator, writer)
                writer.write("\n")
        print(f"Generated {len(self.jobs_ids)} job identifiers.")
        return self.jobs_ids

    def eval_jobs(self, jobs_ids=None, job_ids_file=None, result_filename=None):
        if result_filename is None:
            result_filename = f"jobs_results_{self.model_name}.jsonl"
        # Maybe we need to make some pause
        jobs_results_file_path = os.path.join(self.config.out_folder, result_filename)
        print(f"Evaluating jobs and saving results to {jobs_results_file_path}")
        jobs_awaiting_file_path = os.path.join(
            self.config.out_folder, f"jobs_awaiting_{self.model_name}.jsonl"
        )
        jobs_invalid_file_path = os.path.join(
            self.config.out_folder, f"jobs_invalid_{self.model_name}.jsonl"
        )
        result_file = open(jobs_results_file_path, "w")
        if job_ids_file is not None:
            jobs_ids = read_jsonl(job_ids_file)
        elif jobs_ids is None:
            jobs_ids = self.jobs_ids
        jobs_ids_await = jobs_ids
        n_attempts = 0
        jobs_results = []
        jobs_ids_invalid = []
        # TODO discuss number of attempts and waiting time
        while len(jobs_ids_await) > 0 and n_attempts < 12:
            jobs_ids_await_new = []
            for job_id in jobs_ids_await:
                print(f"Getting result for job {job_id}")
                job_url, conclusion = get_results(job_id, self.config, self.credentials)
                if conclusion == "waiting":
                    jobs_ids_await_new.append(job_id)
                elif conclusion == "error":
                    jobs_ids_invalid.append(job_id)
                else:
                    job_id["url"] = job_url
                    job_id["conclusion"] = conclusion
                    jobs_results.append(job_id)
                    json.dump(job_id, result_file)
                    result_file.write("\n")

            jobs_ids_await = jobs_ids_await_new
            if len(jobs_ids_await) != 0:
                result_file.close()
                save_jsonl(jobs_awaiting_file_path, jobs_ids_await)
                save_jsonl(jobs_invalid_file_path, jobs_ids_invalid)
                print(
                    f"Waiting 360 s to next request of evaluation. {len(jobs_ids_await)} jobs in waiting list."
                )
                time.sleep(360)
                result_file = open(jobs_results_file_path, "a")

            n_attempts += 1

        result_file.close()
        print("Results received")
        print(f"{len(jobs_results)} jobs in results.")
        print(f"{len(jobs_ids_await)} jobs left in waiting list.")
        print(f"{len(jobs_ids_invalid)} jobs are invalid.")
        self.jobs_results = jobs_results
        print(f"Finished evaluating jobs, {len(jobs_results)} jobs processed.")
        return jobs_results

    def get_results(self, job_ids_file=None, result_filename=None):
        # Debugging: Print entry point of the method
        print("Entering get_results method...")

        # If job_ids_file is not provided, set the default file path
        if job_ids_file is None:
            job_ids_file = os.path.join(
                self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
            )
        print(f"Using job_ids_file: {job_ids_file}")

        # Evaluate the jobs using the provided or default job file
        print("Evaluating jobs...")
        self.eval_jobs(job_ids_file=job_ids_file, result_filename=result_filename)

        # If result_filename is not provided, set the default result file name
        if result_filename is None:
            result_filename = f"jobs_results_{self.model_name}.jsonl"
        result_file = os.path.join(self.config.out_folder, result_filename)
        print(f"Using result file: {result_file}")

        # Analyze the results based on the result file
        print("Analyzing results...")
        self.analyze_results(jobs_results_file=result_file)

    def analyze_results(self, jobs_results=None, jobs_results_file=None):
        if jobs_results_file is not None:
            jobs_results = read_jsonl(jobs_results_file)
        if jobs_results is None:
            jobs_results = self.jobs_ids
        print(f"Analyzing results with {len(jobs_results)} job results.")
        results_df = pd.DataFrame(jobs_results)
        total_counts = results_df["conclusion"].value_counts()
        total_ratio = total_counts / len(results_df)
        difficulty_counts = (
            results_df.groupby("difficulty")["conclusion"]
            .value_counts()
            .unstack()
            .fillna(0)
        )
        difficulty_ratios = difficulty_counts.div(difficulty_counts.sum(axis=1), axis=0)

        print("Overall results")
        print(total_counts)
        print("Overall results in ratio")
        print(total_ratio)
        print("Results aggregated by difficulties")
        print(difficulty_counts)
        print("Results in ratios aggregated by difficulties")
        print(difficulty_ratios)

    def eval_dataset(
            self,
            fix_repo_function,
            num_dp: int = None,
            ids_list: List = None,
            force_download=False,
            result_filename=None,
            dataset_folder=None,
    ):
        print("---------------- Downloading data -------------------")
        self.get_dataset(
            num_dp=num_dp,
            force_download=force_download,
            dataset_folder=dataset_folder,
        )
        if ids_list is not None:
            self.dataset = self.dataset.filter(lambda example: filter_by_id(example, ids_list))
        print(f"Got {len(self.dataset)} datapoints")
        print("---------------- Running datapoints -------------------")
        self.run_dataset(fix_repo_function)
        print("---------------- Getting results -------------------")
        self.eval_jobs(result_filename=result_filename)
        self.analyze_results()

    def run_datapoint(self, datapoint, fix_repo_function):
        # This method is for debugging reasons
        jobs_ids_file_path = os.path.join(
            self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
        )
        with open(jobs_ids_file_path, "w") as writer:
            job_identificator = process_datapoint(
                datapoint, fix_repo_function, self.config, self.credentials
            )
            json.dump(job_identificator, writer)
            writer.write("\n")
        print(f"Processed datapoint and wrote job identifier.")
        return job_identificator

    def eval_datapoint(self, job_identificator):
        # This method is for debugging reasons
        pass
# Debugging entry point
if __name__ == "__main__":
    print("Starting CIFixBenchmark...")
    try:
        benchmark = CIFixBenchmark(model_name="your_model_name", config_path=r"C:\Users\Sumt  Kumar\PycharmProjects\lca-baselines\ci-builds-repair\ci-builds-repair-benchmark\config_template.yaml")
        benchmark.eval_dataset(fix_repo_function=fix_apply_diff)
    except Exception as e:
        print(f"An error occurred: {e}")