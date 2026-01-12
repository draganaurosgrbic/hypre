import subprocess
import re
import sys
import os

def get_perf_stats(target_prefix, data_file):
    # Verify file exists before running command
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        return False

    # --group: shows all 5 events side-by-side
    # --no-children: flattens the tree
    # -i: specifies the input data file
    command = ['perf', 'report', '-i', data_file, '--stdio', '--group', '--no-children']
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Regex for 5 columns: 
        # [task-clock] [cycles] [instructions] [L1-miss] [LLC-miss]
        pattern = rf"^\s*([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%.*?({target_prefix}.*)"
        
        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                t_clock = match.group(1) 
                cycles  = match.group(2) 
                instr   = match.group(3) 
                l1      = match.group(4) 
                llc     = match.group(5) 
                symbol  = match.group(6).strip()

                print(f"\nResults for: {symbol}")
                print(f"Source file: {data_file}")
                print("-" * 50)
                print(f"Task-Clock:    {t_clock:>6}% (Time Share)")
                print(f"Cycles:        {cycles:>6}%")
                print(f"Instructions:  {instr:>6}%")
                print(f"L1 Misses:     {l1:>6}%")
                print(f"LLC Misses:    {llc:>6}%")
                
                try:
                    c_val = float(cycles)
                    t_val = float(t_clock)
                    if t_val > 0:
                        intensity = c_val / t_val
                        print(f"Cycle/Clock Intensity: {intensity:>6.2f}")
                    
                    rel_ipc = float(instr) / c_val if c_val > 0 else 0
                    print(f"Relative IPC:          {rel_ipc:>6.2f}")
                except:
                    pass
                return True
                
        print(f"Function {target_prefix} not found in {data_file}.")
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == '__main__':
    # Usage: python parse_perf.py [mode] [perf_data_filename]
    # Example: python parse_perf.py 3 perf_ell8.data
    
    if len(sys.argv) < 2:
        print("Usage: python parse_perf.py [1|2|3] [filename]")
        sys.exit(1)

    mode = sys.argv[1]
    # Default to 'perf.data' if no filename is provided as the second argument
    data_filename = sys.argv[2] if len(sys.argv) >= 3 else "perf.data"
    
    if mode == "3":
        target = "hypre_ELL8_Sequential"
    elif mode == "2":
        target = "hypre_CSRMatrixMatvecTiled7"
    else:
        target = "hypre_CSRMatrixMatvecOutOfPlaceHost"
    
    get_perf_stats(target, data_filename)
