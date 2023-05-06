#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include <signal.h>
#include <time.h>
#include <regex.h>
#include <dirent.h>

#define iter_cores(core_idx, core_count) for(int core_idx = 0; core_idx < core_count; core_idx++)


const char* CPU_FREQ_CORE_PATH = "/sys/devices/system/cpu/";
const char* CPU_FREQ_PATH_EXT = "/cpufreq/cpuinfo_cur_freq";

const char* CPU_UTIL_PATH = "/proc/stat";
const char* NET_UTIL_PATH = "/proc/net/dev";

const char* CPU_OUT_PATH = "/home/pi/cpu_util.ssv";
const char* NET_OUT_PATH = "/home/pi/net_util.ssv";

const char* ETH_IFACE = "eth0";
const char* WIFI_IFACE = "wlan0";

const useconds_t SAMPLE_INTERVAL = 0.1 * 1000000;

pthread_mutex_t read_sync_lock;
pthread_cond_t read_sync_cond;

struct thread_args{
    const char* infile_path;
    const char* outfile_path;
    int (*read_func_ptr)(FILE*, FILE*, bool);
};

volatile sig_atomic_t did_sig_int = false;

size_t cpu_core_count = 0;
char** cpu_core_freq_paths = NULL;

void sig_int_handler(int signal){
    did_sig_int = true;
}

double frac_seconds_since(){
    struct timespec ts;

    timespec_get(&ts, TIME_UTC);
    double float_time = ts.tv_sec + ts.tv_nsec/1000000000.0;
    return float_time;
}

int read_cpu_util(FILE* cpu_in_file, FILE* cpu_out_file, bool is_first_run);

int read_net_util(FILE* net_in_file, FILE* net_out_file, bool is_first_run);

void util_run(const struct thread_args* t_args);

void fetch_cpu_core_freq_paths();