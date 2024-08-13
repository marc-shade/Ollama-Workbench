import subprocess
import json
import os
import streamlit as st
import pandas as pd
from datetime import datetime
import ast
import yaml

def get_conda_list(env_name):
    result = subprocess.run(['conda', 'list', '-n', env_name, '--json'], stdout=subprocess.PIPE, check=True)
    packages = json.loads(result.stdout)
    return {pkg['name']: pkg['version'] for pkg in packages}

def compare_envs(env1, env2):
    env1_packages = get_conda_list(env1)
    env2_packages = get_conda_list(env2)
    
    only_in_env1 = {pkg: ver for pkg, ver in env1_packages.items() if pkg not in env2_packages}
    only_in_env2 = {pkg: ver for pkg, ver in env2_packages.items() if pkg not in env1_packages}
    different_versions = {pkg: (env1_packages[pkg], env2_packages[pkg]) for pkg in env1_packages if pkg in env2_packages and env1_packages[pkg] != env2_packages[pkg]}
    
    return only_in_env1, only_in_env2, different_versions

def find_imports(repo_dir):
    imports = set()
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.add(n.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.level == 0:
                                    imports.add(node.module.split('.')[0])
                    except Exception as e:
                        st.warning(f"Error parsing {file}: {str(e)}")
    return imports

def generate_requirements_with_pipreqs(repo_dir, output_file="requirements.txt"):
    try:
        subprocess.run(['pipreqs', repo_dir, '--force', '--savepath', output_file], check=True)
        with open(output_file, 'r') as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        return f"Error running pipreqs: {e}"
    except FileNotFoundError:
        return "Error: pipreqs not found. Please install it using 'pip install pipreqs'."

def generate_perfect_requirements(repo_dir, working_env):
    repo_imports = find_imports(repo_dir)
    env_packages = get_conda_list(working_env)
    required_packages = {pkg: ver for pkg, ver in env_packages.items() if pkg in repo_imports}
    
    for imp in repo_imports:
        if imp not in required_packages:
            required_packages[imp] = "latest"
    
    requirements = [f"{pkg}=={ver}" if ver != "latest" else pkg for pkg, ver in required_packages.items()]
    return "\n".join(requirements)

def save_comparison(env1, env2, only_in_env1, only_in_env2, different_versions):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{env1}_vs_{env2}_{timestamp}.json"
    
    comparison_data = {
        "env1": env1,
        "env2": env2,
        "only_in_env1": only_in_env1,
        "only_in_env2": only_in_env2,
        "different_versions": different_versions
    }
    
    with open(filename, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    return filename

def load_comparison(filename):
    with open(filename, "r") as f:
        return json.load(f)

def export_env_to_yaml(env_name, yaml_file):
    subprocess.run(['conda', 'env', 'export', '-n', env_name, '-f', yaml_file], check=True)

def convert_yaml_to_requirements(yaml_file, requirements_file):
    with open(yaml_file, 'r') as file:
        env = yaml.safe_load(file)
    
    dependencies = env.get('dependencies', [])
    
    pip_dependencies = []
    for dep in dependencies:
        if isinstance(dep, str):
            package_name = dep.split('=')[0]
            pip_dependencies.append(package_name)
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                package_name = pip_dep.split('=')[0]
                pip_dependencies.append(package_name)
    
    with open(requirements_file, 'w') as file:
        for dep in pip_dependencies:
            file.write(dep + '\n')

st.title("Conda Environment Comparison and Requirements Generator")

env1 = st.text_input("Enter the name of the first Conda environment:")
env2 = st.text_input("Enter the name of the second Conda environment:")
repo_dir = st.text_input("Enter the repository directory:")

if st.button("Compare Environments", key="compare_envs_button"):
    only_in_env1, only_in_env2, different_versions = compare_envs(env1, env2)
    
    st.subheader("Packages only in Environment 1:")
    st.json(only_in_env1)
    
    st.subheader("Packages only in Environment 2:")
    st.json(only_in_env2)
    
    st.subheader("Packages with different versions:")
    st.json(different_versions)
    
    saved_filename = save_comparison(env1, env2, only_in_env1, only_in_env2, different_versions)
    st.success(f"Comparison saved to {saved_filename}")

st.subheader("Load Previous Comparison")
saved_files = [f for f in os.listdir() if f.startswith("comparison_") and f.endswith(".json")]
selected_file = st.selectbox("Select a saved comparison", saved_files)

if st.button("Load Comparison", key="load_comparison_button"):
    loaded_data = load_comparison(selected_file)
    st.json(loaded_data)

st.subheader("Generate requirements.txt")
requirements_method = st.radio("Choose method for generating requirements.txt:", 
                               ("Perfect requirements", "Using pipreqs"))

if requirements_method == "Perfect requirements":
    working_env = st.radio("Select the working environment:", (env1, env2), key="perfect_requirements_env")
    if st.button("Generate Perfect requirements.txt", key="perfect_requirements_button"):
        perfect_requirements = generate_perfect_requirements(repo_dir, working_env)
        st.text_area("Perfect requirements.txt", perfect_requirements, height=300)
        
        if st.button("Save Perfect requirements.txt", key="save_perfect_requirements_button"):
            with open("perfect_requirements.txt", "w") as f:
                f.write(perfect_requirements)
            st.success("perfect_requirements.txt saved successfully!")

elif requirements_method == "Using pipreqs":
    if st.button("Generate requirements.txt using pipreqs", key="pipreqs_requirements_button"):
        pipreqs_requirements = generate_requirements_with_pipreqs(repo_dir)
        st.text_area("Generated requirements.txt (pipreqs)", pipreqs_requirements, height=300)
        
        if st.button("Save pipreqs requirements.txt", key="save_pipreqs_requirements_button"):
            with open("pipreqs_requirements.txt", "w") as f:
                f.write(pipreqs_requirements)
            st.success("pipreqs_requirements.txt saved successfully!")

st.subheader("Environment Comparison Table")
if env1 and env2:
    env1_packages = get_conda_list(env1)
    env2_packages = get_conda_list(env2)
    all_packages = sorted(set(env1_packages.keys()) | set(env2_packages.keys()))
    
    data = []
    for pkg in all_packages:
        data.append({
            "Package": pkg,
            f"{env1} Version": env1_packages.get(pkg, "Not installed"),
            f"{env2} Version": env2_packages.get(pkg, "Not installed")
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df)

st.subheader("Environment Package Search")
search_term = st.text_input("Search for a package:")
if search_term:
    env1_packages = get_conda_list(env1)
    env2_packages = get_conda_list(env2)
    
    results = []
    for pkg, ver in env1_packages.items():
        if search_term.lower() in pkg.lower():
            results.append({"Environment": env1, "Package": pkg, "Version": ver})
    for pkg, ver in env2_packages.items():
        if search_term.lower() in pkg.lower():
            results.append({"Environment": env2, "Package": pkg, "Version": ver})
    
    if results:
        st.table(pd.DataFrame(results))
    else:
        st.warning("No matching packages found.")
