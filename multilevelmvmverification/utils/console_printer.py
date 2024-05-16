import sys
import time


def print_progress_bar(iteration, total, start_time, bar_length=50, prev_percent=-1):
    current_time = time.time()
    elapsed_time = current_time - start_time
    percent = (100 * (iteration / float(total))).__floor__()

    # Update only if percentage changed
    if percent != prev_percent:
        filled_length = min(int(bar_length * iteration // total), bar_length)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        # Estimate time remaining
        if iteration > 0:
            remaining_time = (elapsed_time / iteration) * (total - iteration)
            time_remaining = time.strftime("%H:%M:%S", time.gmtime(remaining_time))
            time_total = time.strftime("%H:%M:%S", time.gmtime(elapsed_time * (total / iteration)))
        else:
            time_remaining = "N/A"
            time_total = "N/A"

        progress_message = f'\rProgress: |{bar}| {percent}% Complete,'
        progress_message += f' Est. time remaining: {time_remaining},'
        progress_message += f' Est. total runtime: {time_total} '

        sys.stdout.write(progress_message)
        sys.stdout.flush()

    return percent
