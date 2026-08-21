<script lang="ts">
  import Button from './Button.svelte';

  let fileInput: HTMLInputElement;
  let previewUrl: string | null = null;
  let selectedFile: File | null = null;
  let status: 'idle' | 'uploading' | 'done' | 'error' = 'idle';
  let errorMessage = '';

  function onFileChange(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    selectedFile = file;
    status = 'idle';
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
  }

  async function upload() {
    if (!selectedFile) return;
    status = 'uploading';
    errorMessage = '';
    try {
      const body = new FormData();
      body.append('file', selectedFile);
      const res = await fetch('/api/garment_image', { method: 'POST', body });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || `Error ${res.status}`);
      }
      status = 'done';
    } catch (e) {
      status = 'error';
      errorMessage = e instanceof Error ? e.message : 'No se pudo subir la prenda';
    }
  }
</script>

<div class="flex flex-col gap-2 rounded border border-gray-300 p-3 dark:border-gray-700">
  <p class="text-sm font-medium">Prenda a probar</p>
  <div class="flex items-center gap-3">
    {#if previewUrl}
      <img src={previewUrl} alt="Prenda seleccionada" class="h-16 w-16 rounded object-cover" />
    {:else}
      <div
        class="flex h-16 w-16 items-center justify-center rounded bg-gray-200 text-xs text-gray-500 dark:bg-gray-700 dark:text-gray-400"
      >
        Sin foto
      </div>
    {/if}
    <div class="flex flex-1 flex-col gap-2">
      <input
        bind:this={fileInput}
        on:change={onFileChange}
        type="file"
        accept="image/*"
        class="text-sm"
      />
      <Button on:click={upload} disabled={!selectedFile || status === 'uploading'} classList="p-1 text-sm">
        {#if status === 'uploading'}
          Subiendo...
        {:else}
          Usar esta prenda
        {/if}
      </Button>
    </div>
  </div>
  {#if status === 'done'}
    <p class="text-sm text-green-600 dark:text-green-400">Prenda lista — ya se está usando en el stream.</p>
  {:else if status === 'error'}
    <p class="text-sm text-red-600 dark:text-red-400">{errorMessage}</p>
  {/if}
</div>
