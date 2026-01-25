import subprocess
import re
import sys
import os

def get_total_duration(data_file):
    try:
        # Extracts the total duration from the perf header
        command = ['perf', 'report', '-i', data_file, '--header', '--stdio']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        match = re.search(r"duration\s+:\s+([0-9.]+)\s+ms", result.stdout)
        if match:
            return float(match.group(1)) / 1000.0
    except Exception:
        pass
    return None

def get_perf_stats(target_prefix, data_file):
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return False

    total_seconds = get_total_duration(data_file)
    # No longer using --group because we only have one event (task-clock)
    command = ['perf', 'report', '-i', data_file, '--stdio', '--no-children']
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # New pattern: Matches the percentage at start of line and the symbol name
        # Example line: "  15.42%  hypre_CSRMatrixMatvecOutOfPlaceHost"
        pattern = rf"^\s*([0-9.]+)%\s+.*?({target_prefix}.*)"
        
        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                t_clock_pct = match.group(1) 
                symbol = match.group(2).strip()

                print(f"\nResults for: {symbol}")
                print(f"Source file: {data_file}")
                print("-" * 50)
                print(f"Task-Clock Share: {t_clock_pct:>7}%")
                
                if total_seconds:
                    # Calculates the actual seconds spent in this kernel
                    kernel_time = total_seconds * (float(t_clock_pct) / 100.0)
                    print(f"Kernel CPU Time:  {kernel_time:>10.4f} seconds")
                    print(f"Total Runtime:    {total_seconds:>10.4f} seconds")
                
                return True
                
        print(f"Function {target_prefix} not found in {data_file}.")
        return False

    except Exception as e:
        print(f"Error parsing perf report: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python parse_perf2.py [1|2|3] [filename]")
        sys.exit(1)

    mode = sys.argv[1]
    data_filename = sys.argv[2]
    
    if mode == "3":
        target = "hypre_ELL8_Sequential"
    elif mode == "2":
        target = "hypre_CSRMatrixMatvecTiled7"
    else:
        target = "hypre_CSRMatrixMatvecOutOfPlaceHost"
    
    get_perf_stats(target, data_filename)
