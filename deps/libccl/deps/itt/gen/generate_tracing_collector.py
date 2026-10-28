from string import Template

tracing_c_body = '''
#include <stdio.h>
#include <dlfcn.h>
#include <ittnotify.h>

__itt_domain* s_domain = NULL;

__attribute__((constructor))
static void initialize_tracing_library()
{
    s_domain = __itt_domain_create("User Tracing Functions");
}

typedef struct _ze_ipc_mem_handle_t
{
    char data[64];              ///< [out] Opaque data representing an IPC handle

} ze_ipc_mem_handle_t;

typedef struct _ze_ipc_event_pool_handle_t
{
    char data[64];              ///< [out] Opaque data representing an IPC handle
} ze_ipc_event_pool_handle_t;


%s
'''

itt_function_body_template = '''
void* $function_name($arguments_with_types)
{
    static __itt_string_handle* str_handle = NULL;
    if (!str_handle)
    {
        str_handle = __itt_string_handle_create("$function_name");
    }

    static void* (*orig_func)($types) = NULL;
    if (orig_func == NULL)
    {
        orig_func = dlsym(RTLD_NEXT, "$function_name");
    }

    void* res = NULL;
    if (orig_func != NULL)
    {
        __itt_task_begin(s_domain, __itt_null, __itt_null, str_handle);
        res = orig_func($arguments);
        __itt_task_end(s_domain);
    }

    return res;
}
'''

itt_void_function_body_template = '''
void $function_name($arguments_with_types)
{
    static __itt_string_handle* str_handle = NULL;
    if (!str_handle)
    {
        str_handle = __itt_string_handle_create("$function_name");
    }

    static void (*orig_func)($types) = NULL;
    if (orig_func == NULL)
    {
        orig_func = dlsym(RTLD_NEXT, "$function_name");
    }

    if (orig_func != NULL)
    {
        __itt_task_begin(s_domain, __itt_null, __itt_null, str_handle);
        orig_func($arguments);
        __itt_task_end(s_domain);
    }

    return;
}
'''

g_function_list = []

functions_list_content = []
with open('./list_of_functions', 'r') as functions_file:
    functions_list_content  = functions_file.readlines()


for line_num in range(len(functions_list_content)):
    line = functions_list_content[line_num]
    if '#' in line[0] or not line.strip():
        continue
    else:
        try:
            return_type, function_name, number_of_arguments = line.split(';')
            if return_type.strip() in ['None', 'none', 'void', '']:
                function_template = itt_void_function_body_template
            else:
                function_template = itt_function_body_template

            # remove all whitespaces
            function_name = function_name.strip()

            arguments = ''
            types = ''
            arguments_with_types = ''

            arg_num = int(number_of_arguments)
            if function_name == "zeMemOpenIpcHandle":
                arguments = 'arg0, arg1, arg2, arg3, arg4'
                types = 'void*, void*, ze_ipc_mem_handle_t, void*, void*'
                arguments_with_types = 'void* arg0, void* arg1, ze_ipc_mem_handle_t arg2, void* arg3, void* arg4'
            elif function_name == "zeEventPoolOpenIpcHandle":
                arguments = 'arg0, arg1, arg2'
                types = 'void*, ze_ipc_event_pool_handle_t, void*'
                arguments_with_types = 'void* arg0, ze_ipc_event_pool_handle_t arg1, void* arg2'
            else:
                if arg_num > 1:
                    arguments = ', '.join(['arg' + str(i) for i in range(arg_num)])
                    types = ', '.join(['void*' for i in range(arg_num)])
                    arguments_with_types = ', '.join(['void* arg' + str(i) for i in range(arg_num)])
                elif arg_num == 1:
                    arguments = 'arg1'
                    types = 'void*'
                    arguments_with_types = 'void* arg1'

            function = Template(function_template).substitute(function_name = function_name.strip(),
                arguments = arguments, types = types, arguments_with_types = arguments_with_types)
            g_function_list.append(function)
        except:
            print("Fail on line %s: %s" % (line_num, line))

all_functions = '\n'.join(g_function_list)
tracing_functions_file_path = './tracing_functions.c'
with open(tracing_functions_file_path, 'w') as tracing_functions_file:
    tracing_functions_file.writelines(tracing_c_body % all_functions)
    print("Source file " + tracing_functions_file_path + " has been generated")
    #print("To compile:")
    #print("gcc tracing_functions.c -D_GNU_SOURCE -fPIC -I$VTUNE_PROFILER_2021_DIR/include -L$VTUNE_PROFILER_2021_DIR/sdk/lib64 -littnotify -lpthread -ldl --shared -o tracing_functions.so")
