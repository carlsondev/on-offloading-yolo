#include "util_fetcher.h"

int read_cpu_core_freqs(int core_count, float* out_core_freq_array){

    if (cpu_core_freq_paths == NULL){
        return 1;
    }

    int ret_val = 0;

    iter_cores(core_idx, cpu_core_count){
        FILE *file_ptr = fopen(cpu_core_freq_paths[core_idx], "r");

        char* freq_str = NULL;
        size_t len = 0;

        getline(&freq_str, &len, file_ptr);

        int freq_val_hz = atoi(freq_str);
        if (freq_val_hz == 0){
            printf("Failed to convert content at path %s to int!", freq_str);
            fclose(file_ptr);
            ret_val |= 1;
            continue;
        }

        out_core_freq_array[core_idx] = ((float) freq_val_hz) / 1000 / 1000;

        fclose(file_ptr);
    }
    return ret_val;
}

int read_cpu_util(FILE* cpu_in_file, FILE* cpu_out_file, bool is_first_run){

    bool do_check_freqs = (cpu_core_freq_paths != NULL);

    if (is_first_run){
        fprintf(cpu_out_file, "seconds cpu user nice system idle iowait irq softirq steal guest guest_nice");

        // If checking frequencies, add those columns
        if (do_check_freqs){
            iter_cores(core_idx, cpu_core_count){
                fprintf(cpu_out_file, " cpu%d_freq", core_idx);
            }
        }

        fprintf(cpu_out_file, "\n");
    }

    // CPU Line 

    char first_line[100];

    fscanf(cpu_in_file, "%[^\n]", first_line);
 
    printf("CPU Line: %s\n", first_line);

    // fprintf is placed in two different places for most accurate seconds
    float out_core_freq_array[cpu_core_count];
    if (do_check_freqs && !read_cpu_core_freqs(cpu_core_count, out_core_freq_array)){
        fprintf(cpu_out_file, "%f %s", frac_seconds_since(), first_line);
        iter_cores(core_idx, cpu_core_count){
            fprintf(cpu_out_file, " %.2f", out_core_freq_array[core_idx]);
        }
    }else{
        fprintf(cpu_out_file, "%f %s", frac_seconds_since(), first_line);
    }
    fprintf(cpu_out_file, "\n");

    freopen(CPU_UTIL_PATH, "r", cpu_in_file);
    return 0;
}

int read_net_util(FILE* net_in_file, FILE* net_out_file, bool is_first_run){

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    int line_count = 0;

    while ((read = getline(&line, &len, net_in_file)) != -1){
        line_count+=1;
        if (line_count <= 2){

            if (!is_first_run){
                continue;
            }
            // Should print given header (2 lines)
            if (line_count == 2){
                fprintf(net_out_file, "seconds %s", line);
            }else{
                fprintf(net_out_file, "%s", line);
            }
            continue;
        }

        // Remove extra padding from the front
        char* first_char_ptr = line;
        while (*first_char_ptr == ' '){
            first_char_ptr++;
        }
        
        if (!strncmp(first_char_ptr, ETH_IFACE, strlen(ETH_IFACE))){
            printf("ETH Line: %s", line);
        }
        if (!strncmp(first_char_ptr, WIFI_IFACE, strlen(WIFI_IFACE))){
            printf("WIFI Line: %s", line);
            fprintf(net_out_file, "%f %s", frac_seconds_since(), line);
        }
        
    }
    rewind(net_in_file);
    return 0;
}

void util_run(const struct thread_args* t_args){
    FILE *file_ptr = fopen(t_args->infile_path, "r");

    if (file_ptr == NULL) return;

    remove(t_args->outfile_path);
    FILE* out_file = fopen(t_args->outfile_path, "a");
    if (out_file == NULL) return;

    pthread_mutex_lock(&read_sync_lock);

    bool is_first_run = true;

    while (true){
	    int ret_val = t_args->read_func_ptr(file_ptr, out_file, is_first_run); 

        if (is_first_run) is_first_run = false;

     	pthread_cond_wait(&read_sync_cond, &read_sync_lock);
    }
    pthread_mutex_unlock(&read_sync_lock);

    fclose(file_ptr);
    fclose(out_file);
}

void fetch_cpu_core_freq_paths(){
    struct dirent *dir_ent;

    DIR *dir_ptr = opendir(CPU_FREQ_CORE_PATH);
    if (dir_ptr == NULL) return;

    regex_t cpu_regex;

    if (regcomp(&cpu_regex, "cpu[0-9]+", REG_EXTENDED)) return;

    int core_count = 0;

    // Count entries that match regex (cpu[:number:])
    while ((dir_ent = readdir(dir_ptr)) != NULL){
        if (regexec(&cpu_regex, dir_ent->d_name, 0, NULL, 0) != REG_NOMATCH){
            core_count+=1;
        }
    }

    // CPU_FREQ_CORE_PATH + CPU_FREQ_PATH_EXT + cpu[0-9] + \0
    int elem_byte_count = strlen(CPU_FREQ_CORE_PATH) + strlen(CPU_FREQ_PATH_EXT) + 4 + 1;
    // CPU_FREQ_CORE_PATH + CPU_FREQ_PATH_EXT + cpu[0-x] + \0
    if (core_count > 9) elem_byte_count += 1;

    // Allocate core_count char pointers
    cpu_core_freq_paths = calloc(core_count, sizeof(char*));

    iter_cores(core_idx, core_count){
        cpu_core_freq_paths[core_idx] = calloc(elem_byte_count, sizeof(char));
        snprintf(cpu_core_freq_paths[core_idx], elem_byte_count,
                "%scpu%d%s", CPU_FREQ_CORE_PATH, core_idx, CPU_FREQ_PATH_EXT); // Copy cpu[0-x] to array

    }

    cpu_core_count = core_count;

    closedir(dir_ptr);
}

int main(int argc, char* argv[]){

    uid_t uid = getuid();
    if (uid == 0){
        printf("Running as root, fetching CPU frequency\n");
        fetch_cpu_core_freq_paths();
    }else{
        printf("Not running as root, will not fetch CPU frequency\n");
    }

    

    signal(SIGINT, sig_int_handler);

    pthread_t cpu_util_thread, net_util_thread;

    pthread_mutex_init(&read_sync_lock, NULL);
    pthread_cond_init(&read_sync_cond, NULL);

    struct thread_args cpu_util_args = {.infile_path=CPU_UTIL_PATH,
                                        .outfile_path=CPU_OUT_PATH,
                                        .read_func_ptr=&read_cpu_util};
    struct thread_args net_util_args = {.infile_path=NET_UTIL_PATH,
                                        .outfile_path=NET_OUT_PATH,
                                        .read_func_ptr=&read_net_util};

    int ret = pthread_create(&cpu_util_thread, NULL, (void*) &util_run, &cpu_util_args);
    pthread_create(&net_util_thread, NULL, (void*) &util_run, &net_util_args);
    
    while (!did_sig_int){

        // Every READ_INTERVAL seconds, broadcast the condition to all threads
        pthread_mutex_lock(&read_sync_lock);

        pthread_cond_broadcast(&read_sync_cond);

        pthread_mutex_unlock(&read_sync_lock);

        usleep(SAMPLE_INTERVAL);

    }

    pthread_kill(cpu_util_thread, 0);
    pthread_kill(net_util_thread, 0);

    if (cpu_core_freq_paths != NULL){
        iter_cores(core_idx, cpu_core_count){
            if (cpu_core_freq_paths[core_idx] != NULL)
                free(cpu_core_freq_paths[core_idx]);
        }
        free(cpu_core_freq_paths);
    }

    return 0;
}
