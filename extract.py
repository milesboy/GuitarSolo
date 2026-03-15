import subprocess
import sys
import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "downloads")
COOKIES_FILE = os.path.join(PROJECT_DIR, "cookies.txt")


def find_ffmpeg():
    """Find ffmpeg binary — check PATH first, then imageio_ffmpeg bundle."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def find_yt_dlp():
    """Find yt-dlp — check PATH first, then try as Python module."""
    path = shutil.which("yt-dlp")
    if path:
        return [path]
    return [sys.executable, "-m", "yt_dlp"]


def extract_audio(url):
    """Download and extract audio from a YouTube URL using yt-dlp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        print("Error: ffmpeg not found. Install it or run: pip install imageio-ffmpeg")
        sys.exit(1)

    # yt-dlp expects ffmpeg.exe in a directory — create a symlink if needed
    ffmpeg_dir = os.path.join(PROJECT_DIR, ".ffmpeg")
    ffmpeg_link = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_link):
        os.makedirs(ffmpeg_dir, exist_ok=True)
        shutil.copy2(ffmpeg_path, ffmpeg_link)

    yt_dlp_cmd = find_yt_dlp()
    output_template = os.path.join(OUTPUT_DIR, "%(title)s.%(ext)s")

    if not os.path.exists(COOKIES_FILE):
        print(f"Error: cookies.txt not found at {COOKIES_FILE}")
        print("Export your YouTube cookies using a browser extension (Get cookies.txt LOCALLY)")
        print("and save the file as cookies.txt in the project folder.")
        sys.exit(1)

    cmd = [
        *yt_dlp_cmd,
        "--ffmpeg-location", ffmpeg_dir,
        "--no-playlist",
        "--cookies", COOKIES_FILE,
        "--remote-components", "ejs:github",
        "-x",
        "--audio-format", "wav",
        "-o", output_template,
        url,
    ]

    print(f"Downloading audio from: {url}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Error downloading audio.")
        sys.exit(1)

    # Find the most recent wav in downloads
    wav_files = [
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".wav")
    ]
    if wav_files:
        return max(wav_files, key=os.path.getmtime)

    print("Could not find downloaded audio file.")
    sys.exit(1)


def get_library():
    """Return list of downloaded audio files."""
    if not os.path.exists(OUTPUT_DIR):
        return []
    extensions = (".wav", ".webm", ".mp3", ".m4a", ".ogg")
    files = [
        os.path.join(OUTPUT_DIR, f)
        for f in sorted(os.listdir(OUTPUT_DIR))
        if f.lower().endswith(extensions)
    ]
    return files


def pick_song():
    """Show library and let user pick a song or download a new one."""
    library = get_library()

    print("\n=== GuitarSolo ===\n")

    if library:
        print("Your library:")
        for i, filepath in enumerate(library, 1):
            name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"  {i}. {name}")
        print(f"\n  N. Download a new song\n")

        choice = input("Pick a number or N for new: ").strip().lower()

        if choice == "n":
            url = input("Paste a YouTube URL: ").strip()
            if not url:
                print("No URL provided. Exiting.")
                sys.exit(1)
            return extract_audio(url)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(library):
                return library[idx]
        except ValueError:
            pass

        print("Invalid choice.")
        sys.exit(1)
    else:
        print("No songs in library yet.\n")
        url = input("Paste a YouTube URL for the song you want to learn: ").strip()
        if not url:
            print("No URL provided. Exiting.")
            sys.exit(1)
        return extract_audio(url)


def main():
    filepath = pick_song()
    print(f"\nSelected: {os.path.basename(filepath)}")
    print("Ready for analysis!")


if __name__ == "__main__":
    main()
