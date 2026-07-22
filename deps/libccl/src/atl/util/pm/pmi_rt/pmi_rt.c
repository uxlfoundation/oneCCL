#include "util/pm/codec/pm_rt_codec.h"

#include <stdlib.h>

#include "pmi/pmi.h"

#include "util/pm/pm_rt.h"

#define PMI_RT_KEY_FORMAT "%s-%d"

typedef struct pmi_pm_rt_context {
    pm_rt_desc_t pmrt_desc;
    struct {
        int initialized;
        int ref_cnt;
        int max_keylen;
        int max_vallen;
        char *key_storage;
        char *val_storage;
        char *kvsname;
    } pmirt_main;
} pmi_pm_context_t;

/* Ensures that this is allocated/initialized only once per process */
static pmi_pm_context_t pmi_ctx_singleton;

static void pmirt_finalize(pm_rt_desc_t *pmrt_desc) {
    pmi_pm_context_t *ctx = container_of(pmrt_desc, pmi_pm_context_t, pmrt_desc);
    if (!ctx->pmirt_main.initialized)
        return;

    if (--ctx->pmirt_main.ref_cnt)
        return;

    free(ctx->pmirt_main.kvsname);
    free(ctx->pmirt_main.key_storage);
    free(ctx->pmirt_main.val_storage);

    PMI_Finalize();

    memset(ctx, 0, sizeof(*ctx));
}

static atl_status_t pmirt_kvs_put(pm_rt_desc_t *pmrt_desc,
                                  char *kvs_key,
                                  int proc_idx,
                                  const void *kvs_val,
                                  size_t kvs_val_len) {
    int ret;
    pmi_pm_context_t *ctx = container_of(pmrt_desc, pmi_pm_context_t, pmrt_desc);

    if (!ctx->pmirt_main.initialized)
        return ATL_STATUS_FAILURE;

    if ((int)kvs_val_len > ctx->pmirt_main.max_vallen)
        return ATL_STATUS_FAILURE;

    ret = snprintf(ctx->pmirt_main.key_storage,
                   ctx->pmirt_main.max_keylen,
                   PMI_RT_KEY_FORMAT,
                   kvs_key,
                   proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = encode(kvs_val, kvs_val_len, ctx->pmirt_main.val_storage, ctx->pmirt_main.max_vallen);
    if (ret)
        return ATL_STATUS_FAILURE;

    ret = PMI_KVS_Put(
        ctx->pmirt_main.kvsname, ctx->pmirt_main.key_storage, ctx->pmirt_main.val_storage);
    if (ret != PMI_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMI_KVS_Commit(ctx->pmirt_main.kvsname);
    if (ret != PMI_SUCCESS)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

static atl_status_t pmirt_kvs_get(pm_rt_desc_t *pmrt_desc,
                                  char *kvs_key,
                                  int proc_idx,
                                  void *kvs_val,
                                  size_t kvs_val_len) {
    int ret;
    pmi_pm_context_t *ctx = container_of(pmrt_desc, pmi_pm_context_t, pmrt_desc);

    if (!ctx->pmirt_main.initialized)
        return ATL_STATUS_FAILURE;

    ret = snprintf(ctx->pmirt_main.key_storage,
                   ctx->pmirt_main.max_keylen,
                   PMI_RT_KEY_FORMAT,
                   kvs_key,
                   proc_idx);
    if (ret < 0)
        return ATL_STATUS_FAILURE;

    ret = PMI_KVS_Get(ctx->pmirt_main.kvsname,
                      ctx->pmirt_main.key_storage,
                      ctx->pmirt_main.val_storage,
                      ctx->pmirt_main.max_vallen);

    if (ret != PMI_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = decode(ctx->pmirt_main.val_storage, kvs_val, kvs_val_len);
    if (ret)
        return ATL_STATUS_FAILURE;

    return ATL_STATUS_SUCCESS;
}

static void pmirt_barrier(pm_rt_desc_t *pmrt_desc) {
    pmi_pm_context_t *ctx = container_of(pmrt_desc, pmi_pm_context_t, pmrt_desc);

    if (!ctx->pmirt_main.initialized)
        return;

    (void)PMI_Barrier();
}

atl_status_t pmirt_update(int *proc_idx, int *proc_count) {
    PMI_Get_size((int *)proc_idx);
    PMI_Get_rank((int *)proc_count);
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmirt_wait_notification() {
    return ATL_STATUS_FAILURE;
}

pm_rt_ops_t ops = {
    .finalize = pmirt_finalize,
    .barrier = pmirt_barrier,
    .update = pmirt_update,
    .wait_notification = pmirt_wait_notification,
};

pm_rt_kvs_ops_t kvs_ops = {
    .put = pmirt_kvs_put,
    .get = pmirt_kvs_get,
};

atl_status_t pmirt_init(int *proc_idx, int *proc_count, pm_rt_desc_t **pmrt_desc) {
    int ret, spawned, max_kvsnamelen;
    int proc_idx_tmp, proc_count_tmp;

    if (pmi_ctx_singleton.pmirt_main.initialized) {
        PMI_Get_size((int *)proc_idx);
        PMI_Get_rank((int *)proc_count);
        *pmrt_desc = &pmi_ctx_singleton.pmrt_desc;
        pmi_ctx_singleton.pmirt_main.ref_cnt++;
        return ATL_STATUS_SUCCESS;
    }

    ret = PMI_Init(&spawned);
    if (ret != PMI_SUCCESS)
        return ATL_STATUS_FAILURE;

    ret = PMI_Get_size(&proc_count_tmp);
    if (ret != PMI_SUCCESS)
        goto err_pmi;
    *proc_count = proc_count_tmp;
    ret = PMI_Get_rank(&proc_idx_tmp);
    if (ret != PMI_SUCCESS)
        goto err_pmi;
    *proc_idx = proc_idx_tmp;

    ret = PMI_KVS_Get_name_length_max(&max_kvsnamelen);
    if (ret != PMI_SUCCESS)
        goto err_pmi;

    pmi_ctx_singleton.pmirt_main.kvsname = (char *)calloc(1, max_kvsnamelen);
    if (!pmi_ctx_singleton.pmirt_main.kvsname)
        goto err_pmi;

    ret = PMI_KVS_Get_my_name(pmi_ctx_singleton.pmirt_main.kvsname, max_kvsnamelen);
    if (ret != PMI_SUCCESS)
        goto err_alloc_key;

    ret = PMI_KVS_Get_key_length_max(&pmi_ctx_singleton.pmirt_main.max_keylen);
    if (ret != PMI_SUCCESS)
        goto err_alloc_key;

    pmi_ctx_singleton.pmirt_main.key_storage =
        (char *)calloc(1, pmi_ctx_singleton.pmirt_main.max_keylen);
    if (!pmi_ctx_singleton.pmirt_main.key_storage)
        goto err_alloc_key;

    ret = PMI_KVS_Get_value_length_max(&pmi_ctx_singleton.pmirt_main.max_vallen);
    if (ret != PMI_SUCCESS)
        goto err_alloc_val;

    pmi_ctx_singleton.pmirt_main.val_storage =
        (char *)calloc(1, pmi_ctx_singleton.pmirt_main.max_vallen);
    if (!pmi_ctx_singleton.pmirt_main.val_storage)
        goto err_alloc_val;

    pmi_ctx_singleton.pmirt_main.initialized = 1;
    pmi_ctx_singleton.pmirt_main.ref_cnt = 1;
    pmi_ctx_singleton.pmrt_desc.ops = &ops;
    pmi_ctx_singleton.pmrt_desc.kvs_ops = &kvs_ops;
    *pmrt_desc = &pmi_ctx_singleton.pmrt_desc;

    return ATL_STATUS_SUCCESS;
err_alloc_val:
    free(pmi_ctx_singleton.pmirt_main.key_storage);
err_alloc_key:
    free(pmi_ctx_singleton.pmirt_main.kvsname);
err_pmi:
    PMI_Finalize();
    return ATL_STATUS_FAILURE;
}
