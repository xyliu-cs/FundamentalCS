#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

// 4096 entries in inverted swap table
// total pages from global mem (128KB/32B)
#define ENTRY_NUMER (1<<12)
// the first bit of this unsigned 32-bit integer is set to 1
#define EMPTY_ENTRY u32(1<<31)
// total entries within one page (32B/1B)
#define ENTRY_PER_PAGE (1<<5)




// implement swap table in the global memory (disk)
// store virtual page number and corresponding disk number
__device__ __managed__ u32 swap_table[ENTRY_NUMER * 2];


__device__ void init_invert_page_table(VirtualMemory *vm) {
  // page table does not include the 5-bit offset
  // the lower
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = EMPTY_ENTRY; // invalid := MSB is 1

    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i; // corresponding frame index

    vm->invert_page_table[i + vm->PAGE_ENTRIES*2] = EMPTY_ENTRY; // times after the last reference

  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);

  // initialize swap table entry's MSB to 1
  for (int i = 0; i < ENTRY_NUMER; i++) {
    swap_table[i] = EMPTY_ENTRY;
  }
  
}


/* read a single element from data buffer */
__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {

  unsigned offset = addr & 0x1f;                   // extract offset from addr
  u32 page_number = (addr >> 5) & ((1 << 13) - 1); // extract the page number from addr

  uchar read_val;           // return value
  int found_epty_entry = 0; //flag: set to 1 if found an empty entry
  int page_hit = 0;         //flag: set to 1 if page hits

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    u32 entry = vm->invert_page_table[i];

    if (entry == EMPTY_ENTRY) 
      found_epty_entry = 1;

    else if (entry == page_number) {
      page_hit = 1;

      int frame_number = vm->invert_page_table[i + vm->PAGE_ENTRIES];
      int mem_addr = (frame_number << 5) + offset;
      read_val = vm->buffer[mem_addr];

      for (int j = 0; j < vm->PAGE_ENTRIES; j++) {
        if ((j != i) && (vm->invert_page_table[j + vm->PAGE_ENTRIES *2] != EMPTY_ENTRY))
          vm->invert_page_table[j + vm->PAGE_ENTRIES *2] += 1;
      }
    }
  }

  if (page_hit)
    return read_val;
  // ------------------------- function may end here -------------------------
  
  assert(!page_hit && !found_epty_entry);
  // Not found in the page table, search in the swap table
  // If not in the swap table, return an error
  // If found in the swap table, replace the LRU and swap in that page
  // update page table and swap table
  
  // page fault
  (*vm->pagefault_num_ptr)++; 
  
  // find LRU
  // start at index 0
  u32 LRU_VPN = vm->invert_page_table[0];
  int LRU_index = vm->invert_page_table[vm->PAGE_ENTRIES];
  int LRU_fqcy = vm->invert_page_table[vm->PAGE_ENTRIES * 2];

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] > LRU_fqcy) {
      LRU_VPN = vm->invert_page_table[i];
      LRU_index = vm->invert_page_table[i + vm->PAGE_ENTRIES];
      LRU_fqcy = vm->invert_page_table[i + vm->PAGE_ENTRIES * 2];
    }
  }

  // flags: set to 1 if found
  int found_epty_frame = 0;
  int found_mtch_frame = 0;
  // address: empty disk frame index
  int empty_frame;
  int match_frame;
  // memeory and disk real address
  int mem, mem_target, disk, disk_empty;

  // look up in swap table
  for (int i=0; i < ENTRY_NUMER; i++) {
    if (swap_table[i] == page_number) {
      found_mtch_frame = 1;
      match_frame = i;
      swap_table[i] = EMPTY_ENTRY;
      break; 
    }    
  }

  for (int i=0; i < ENTRY_NUMER; i++) {
    if (swap_table[i] == EMPTY_ENTRY) {
      found_epty_frame = 1;
      empty_frame = i;
      break; 
    }    
  }

  // if not exist in swap table, raise an error
  assert(found_epty_frame);
  assert(found_mtch_frame);

  uchar tmp_buffer;
  disk = match_frame << 5;
  disk_empty = empty_frame << 5;
  mem = LRU_index << 5;
  mem_target = mem + offset;

  // [MEM TO DISK]: LRU page to empty disk frame
  // [DISK TO MEM]: matched disk frame to LRU memory page
  for (int i=0; i < ENTRY_PER_PAGE; i++) {
    tmp_buffer = vm->buffer[mem + i];
    vm->buffer[mem + i] = vm->storage[disk + i];
    vm->storage[disk_empty + i] = tmp_buffer;
  }

  // read a single entry from memory
  read_val = vm->buffer[mem_target];

  // update page table, swap table
  vm->invert_page_table[LRU_index] = page_number;
  vm->invert_page_table[LRU_index + vm->PAGE_ENTRIES *2] = 0;
  swap_table[empty_frame] = LRU_VPN;

  return read_val; 
}


/* write a single element to data buffer */
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {

  unsigned offset = addr & 0x1f;                    // extract offset from addr
  u32 page_number = (addr >> 5) & ((1 << 13) - 1);  // extract the page number from addr 
  
  // flags: set to 1 if found 
  int page_hit = 0; 
  int found_empty_page = 0;

  int frame_number = 0; // 0~1023

  // look up the page table for the frame number
  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    u32 entry = vm->invert_page_table[i];
    // page hit
   if (entry == page_number) {
      page_hit = 1;
      frame_number = vm->invert_page_table[i + vm->PAGE_ENTRIES];
      assert(frame_number == i);
      int mem_addr = (frame_number << 5) + offset;
      vm->buffer[mem_addr] = value;                         // write to memory

      for (int j = 0; j < vm->PAGE_ENTRIES; j++) {
        if ((j != i) && (vm->invert_page_table[j + vm->PAGE_ENTRIES *2] != EMPTY_ENTRY))
          vm->invert_page_table[j + vm->PAGE_ENTRIES *2] += 1;
      }
      // vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1; // reference + 1
      // continue looping to ensure we search the whole array to find empty entry
      return;
    }
  }

  // ------------------------- function may end here ------------------------- 
  
  assert(!page_hit);

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    u32 entry = vm->invert_page_table[i];
    if (entry == EMPTY_ENTRY) {
      found_empty_page = 1;
      (*vm->pagefault_num_ptr)++;                                    // page fault
      vm->invert_page_table[i] = page_number;
      vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0;           //update page table
      int mem_addr = (i << 5) + offset;
      vm->buffer[mem_addr] = value;                                  //write to memory
      return;
    }
  }

  // ------------------------- function may end here -------------------------
  
  assert(!page_hit && !found_empty_page);
  // Not found in the page table and page table is full (physical mem is full)
  // the page to swap out from the PT is the least indexed LRU
  // determine which page to swap in
  // look up the vm addr in the swap table first: 
  // if found, swap in that frame; else, swap in an available frame from disk

  // page fault
  (*vm->pagefault_num_ptr)++; 
  
  // find LRU
  // start at index 0
  u32 LRU_VPN = vm->invert_page_table[0];
  int LRU_index = vm->invert_page_table[vm->PAGE_ENTRIES];
  int LRU_fqcy = vm->invert_page_table[vm->PAGE_ENTRIES * 2];

  for (int i = 1; i < vm->PAGE_ENTRIES; i++) {
    if (vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] > LRU_fqcy) {
      LRU_VPN = vm->invert_page_table[i];
      LRU_index = vm->invert_page_table[i + vm->PAGE_ENTRIES];
      LRU_fqcy = vm->invert_page_table[i + vm->PAGE_ENTRIES * 2];
    }
  }


  // flags: set to 1 if found
  int found_epty_frame = 0;
  int found_swap_frame = 0;
  // address: matching & empty disk frame index
  int empty_frame = 0;
  int match_frame = 0;

  // search in swap table
  for (int i=0; i < ENTRY_NUMER; i++) {
    // found matching entry
    if (swap_table[i] == page_number) {
      found_swap_frame = 1;
      match_frame = i;
      swap_table[i] = EMPTY_ENTRY; //after loading, disk frame becomes available
      break;
    }  
  }

  for (int i = 0; i < ENTRY_NUMER; i++)
  {
    if (swap_table[i] == EMPTY_ENTRY) {
      found_epty_frame = 1;
      empty_frame = i;
      break;
    }   
  }
  
  assert(found_epty_frame);


  uchar tmp_buffer;
  int mem_page_base = LRU_index << 5;
  int disk_empty_base = empty_frame << 5;
  int disk_target_base = match_frame << 5;
  int mem_addr = mem_page_base + offset;

  // [MEM TO DISK]: Copy the whole LRU page
  // [DISK TO MEM]: Only if found matching frame (have some data to copy);
  // otherwise just cleared memory page
  for (int i=0; i < ENTRY_PER_PAGE; i++) {
    tmp_buffer = vm->buffer[mem_page_base + i]; //LRU to tmp
    if (found_swap_frame)
      vm->buffer[mem_page_base + i] = vm->storage[disk_target_base + i]; // matched disk frame to LRU
    else
      vm->buffer[mem_page_base + i] = 0;  //clear memory LRU page 
    vm->storage[disk_empty_base + i] = tmp_buffer; // tmp to empty disk frame
  }
  
  // write a single entry
  vm->buffer[mem_addr] = value;

  // update page table & swap table
  vm->invert_page_table[LRU_index] = page_number;
  vm->invert_page_table[LRU_index + vm->PAGE_ENTRIES*2] = 0;
  swap_table[empty_frame] = LRU_VPN;
  
  return;

}

/* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {

  	for (int i = 0; i < input_size; i++) {
		  results[i] = vm_read(vm, i+offset);
	  }

}

