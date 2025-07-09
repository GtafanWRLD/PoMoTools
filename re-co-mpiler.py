import os
import polib
import sys

def mo_to_po(mo_path, po_path):
    """Convert a single .mo file to .po file"""
    try:
        mo_entries = polib.mofile(mo_path)
        po = polib.POFile()
        for entry in mo_entries:
            po.append(entry)
        po.metadata = mo_entries.metadata
        po.save(po_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to decompile '{mo_path}': {e}")
        return False

def po_to_mo(po_path, mo_path):
    """Convert a single .po file to .mo file"""
    try:
        po = polib.pofile(po_path)
        po.save_as_mofile(mo_path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to compile '{po_path}': {e}")
        return False

def decompile_folder(mo_folder, po_folder):
    """Decompile all .mo files in a folder to .po files"""
    if not os.path.exists(po_folder):
        os.makedirs(po_folder)
    
    mo_files = [f for f in os.listdir(mo_folder) if f.endswith('.mo')]
    if not mo_files:
        print("No .mo files found in the specified folder.")
        return
    
    print(f"Found {len(mo_files)} .mo files to decompile...")
    success_count = 0
    
    for filename in mo_files:
        mo_path = os.path.join(mo_folder, filename)
        po_filename = os.path.splitext(filename)[0] + '.po'
        po_path = os.path.join(po_folder, po_filename)
        print(f"Decompiling: {filename} → {po_filename} ... ", end="", flush=True)
        if mo_to_po(mo_path, po_path):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\nDecompilation complete: {success_count}/{len(mo_files)} files processed successfully.")

def compile_folder(po_folder, mo_folder):
    """Compile all .po files in a folder to .mo files"""
    if not os.path.exists(mo_folder):
        os.makedirs(mo_folder)
    
    po_files = [f for f in os.listdir(po_folder) if f.endswith('.po')]
    if not po_files:
        print("No .po files found in the specified folder.")
        return
    
    print(f"Found {len(po_files)} .po files to compile...")
    success_count = 0
    
    for filename in po_files:
        po_path = os.path.join(po_folder, filename)
        mo_filename = os.path.splitext(filename)[0] + '.mo'
        mo_path = os.path.join(mo_folder, mo_filename)
        print(f"Compiling: {filename} → {mo_filename} ... ", end="", flush=True)
        if po_to_mo(po_path, mo_path):
            print("OK")
            success_count += 1
        else:
            print("FAILED")
    
    print(f"\nCompilation complete: {success_count}/{len(po_files)} files processed successfully.")

def detect_file_types(folder_path):
    """Detect what file types are in the folder"""
    mo_count = len([f for f in os.listdir(folder_path) if f.endswith('.mo')])
    po_count = len([f for f in os.listdir(folder_path) if f.endswith('.po')])
    return mo_count, po_count

def main():
    print("=" * 60)
    print("           PO/MO File Converter Tool")
    print("=" * 60)
    
    # Choose action first
    print("\nWhat would you like to do?")
    print("1. Decompile .mo files to .po files")
    print("2. Compile .po files to .mo files")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Please enter 1 or 2.")
    
    action = 'decompile' if choice == '1' else 'compile'
    file_extension = '.mo' if action == 'decompile' else '.po'
    
    # Get input folder
    while True:
        folder_path = input(f"\nEnter the folder path with your {file_extension} files: ").strip().strip('"')
        if os.path.isdir(folder_path):
            break
        print("Error: Folder does not exist. Please try again.")
    
    # Get script directory (where the .py/.exe file is located)
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # Check if files exist
    if action == 'decompile':
        mo_count = len([f for f in os.listdir(folder_path) if f.endswith('.mo')])
        if mo_count == 0:
            print("\nNo .mo files found in the specified folder.")
            input("Press Enter to exit...")
            return
        print(f"\nFound {mo_count} .mo files to decompile.")
    else:
        po_count = len([f for f in os.listdir(folder_path) if f.endswith('.po')])
        if po_count == 0:
            print("\nNo .po files found in the specified folder.")
            input("Press Enter to exit...")
            return
        print(f"\nFound {po_count} .po files to compile.")
    
    # Perform the action
    print(f"\nStarting {action}...")
    
    if action == 'decompile':
        output_folder = os.path.join(script_dir, 'decompiled')
        decompile_folder(folder_path, output_folder)
        print(f"\nDecompiled .po files are saved in: {output_folder}")
    else:
        output_folder = os.path.join(script_dir, 'compiled')
        compile_folder(folder_path, output_folder)
        print(f"\nCompiled .mo files are saved in: {output_folder}")
    
    print("\nAll done!")
    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)