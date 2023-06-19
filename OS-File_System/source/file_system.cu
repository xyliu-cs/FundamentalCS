#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// the MSB is set to 1 for the 16-bit number
#define INVALID_MSB_16 1 << 15
#define All_8_bits_are_1 uchar((1 << 8) - 1)

__device__ __managed__ u32 gtime = 0;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

  /*
    volume 1 byte per entry

    [VCB]: volume[0:4095], 4KB, 32K places, 1 place (1 bit) per block;
           For each block: 0 is empty, 1 is full.

    [FCB]: volume[4096:36863], 32KB, 1024K places, 1 place (32B) per file
           In each place:
           Index    Size     Valid Range           Item
           0-19     20 B      NA                 Filename
           20-23    2 B       0~4K*1024KB-1      File Size (0-1024KB)
           24-25    4 B       0~64K-1            File starting block index
           26-27    2 B       0~64K-1            Created Time
           28-29    2 B       0~64K-1            Modified Time
           30       1 B
           31       1 B       0~256              Valid Bit (0/1)

    [File]: volume[36864:1085439], 1024KB, 32K blocks, 1 block (32B) per file

  */

  // init every volume entry to 0
  for (int i = 0; i < fs->STORAGE_SIZE; i++)
  {
    fs->volume[i] = 0;
  }
}

/* ----------- Some helper functions are defined below ----------- */

// erase the corresponding bits to 0 in VCB
// mode = 0: erase to 0; mode = 1, set to 1
__device__ void set_file_bits(FileSystem *fs, u16 start, u16 b_len, int mode)
{
  u32 end = start + b_len - 1;
  u32 start_row = start / 8;
  u16 start_offset = start % 8;
  u32 end_row = end / 8;
  u16 end_offset = end % 8;
  const uchar tmp_1 = 1;

  if (start_row == end_row)
  {
    uchar target = fs->volume[start_row];

    for (u16 i = start_offset; i <= end_offset; i++)
    {
      if (mode == 0)
      {
        target = target & (~(tmp_1 << i));
      }
      else if (mode == 1)
      {
        target = target | (tmp_1 << i);
      }
    }
    fs->volume[start_row] = target;
    return;
  }

  else
  {
    if (mode == 0)
    {
      fs->volume[start_row] = fs->volume[start_row] & ((1 << start_offset) - 1);
      for (int i = start_row + 1; i < end_row; i++)
      {
        fs->volume[i] = 0;
      }
      uchar tmp = ((1 << 8) - 1) - ((1 << (end_offset + 1)) - 1); // 11110000
      fs->volume[end_row] = fs->volume[end_row] & tmp;
    }

    else if (mode == 1)
    {
      fs->volume[start_row] |= ~((tmp_1 << start_offset) - 1);
      for (int i = start_row + 1; i < end_row; i++)
      {
        fs->volume[i] = All_8_bits_are_1;
      }
      fs->volume[end_row] |= ((tmp_1 << (end_offset + 1)) - 1);
    }

    return;
  }
}

// Count number of zeros in FCB
__device__ int count_zeros(FileSystem *fs)
{

  int count;
  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++)
  {
    uchar bitsrow = fs->volume[i];
    if (bitsrow == 0)
    {
      count += 8;
      continue;
    }

    for (int idx = 0; idx < 8; idx++)
    {
      const uchar tmp_1 = 1;
      if ((bitsrow & (tmp_1 << idx)) == 0)
      {
        count++;
      }
    }
  }
  // printf("# of zeros in VCB = %d\n", count);
  return count;
}

// bits operation helper function
// flag = 1, assign, flag = 0, do not assign
__device__ u16 get_0_bit_in_VCB(FileSystem *fs, int set)
{

  for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++)
  {
    uchar bitsrow = fs->volume[i];
    if (bitsrow != All_8_bits_are_1)
    {
      // Masking to find 0 in 8 bits
      for (int idx = 0; idx < 8; idx++)
      {
        if ((bitsrow & (1 << idx)) == 0)
        {
          if (set)
          {
            fs->volume[i] = (bitsrow | (1 << idx)); // assign here
          }
          // printf("hit, index = %d\n", i * 8 + idx);
          return (i * 8 + idx);
        }
      }
    }
  }
  // should never reach here
  return INVALID_MSB_16;
}

// Function to implement strcmp function
__device__ int _strcmp(const char *s1, const char *s2)
{
  while (*s1)
  {
    // if characters differ, or end of the second string is reached
    if (*s1 != *s2)
    {
      break;
    }
    s1++;
    s2++;
  }
  // return the ASCII difference
  return *(const uchar *)s1 - *(const uchar *)s2;
}

// Function to implement strcpy function
__device__ char* _strcpy(char *destination, const char *source)
{
  // return if no memory is allocated to the destination
  if (destination == NULL)
  {
    return NULL;
  }
  char *ptr = destination;
  while (*source != '\0')
  {
    *destination = *source;
    destination++;
    source++;
  }
  // include the terminating null character
  *destination = '\0';
  return ptr;
}

__device__ u32 _strlen(const char *s)
{
  u32 count = 0;
  while (*s != '\0')
  {
    count++;
    s++;
  }
  return count;
}

/* -------------------------- End Of Helper Functions -------------------------*/

/* Implement open operation here */
/* Return FCB Index from 0 to 1024 */
__device__ u32 fs_open(FileSystem *fs, char *filename, int op)
{
  // checkpoints
  int valid_filename_size = 1;
  int valid_operand = 0;

  if (_strlen(filename) > fs->MAX_FILENAME_SIZE)
    valid_filename_size = 0;
  assert(valid_filename_size);

  if ((op == G_WRITE) || (op == G_READ))
    valid_operand = 1;
  assert(valid_operand);

  // read mode
  int found_target_FCB_in_read = 0;
  if (op == G_READ)
  {
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if (fs->volume[FCB_base + 31] != 0)
      {
        if (_strcmp(filename, (char *)&(fs->volume[FCB_base])) == 0)
        {
          found_target_FCB_in_read = 1;
          return i;
        }
      }
    }
    printf("filename is %s\n", filename);
    assert(found_target_FCB_in_read);
  }

  else if (op == G_WRITE)
  {
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if (fs->volume[FCB_base + 31] != 0)
      {
        if (_strcmp(filename, (char *)&(fs->volume[FCB_base])) == 0)
          return i;
      }
    }
    /* Not found in the FCB, then create a new 0 byte file
       Add an entry in FCB and take a empty place in VCB  */
    // Search & Change on VCB (only need 1 place)
    u16 ret_index = get_0_bit_in_VCB(fs, 1);
    if (ret_index == INVALID_MSB_16)
    {
      printf("Could not get an empty space in VCB\n");
      printf("filename is %s\n", filename);
      return 0;
    }


    // checkpoints
    int found_valid_FCB = 0;
    // Change on FCB
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if ((fs->volume[FCB_base + 31]) == 0)
      {
        found_valid_FCB = 1;

        // clear this FCB
        for (int j = 0; j < 32; j++)
        {
          fs->volume[FCB_base + j] = 0;
        }

        // copy file name
        _strcpy((char *)&(fs->volume[FCB_base]), filename);
        // set file size (0 byte)
        *((u32 *)&(fs->volume[FCB_base + 20])) = 0;
        // set block index
        *((u16 *)&(fs->volume[FCB_base + 24])) = ret_index;
        // set create time
        fs->volume[FCB_base + 26] = gtime / 256;
        fs->volume[FCB_base + 27] = gtime % 256;
        // set modified time
        fs->volume[FCB_base + 28] = gtime / 256;
        fs->volume[FCB_base + 29] = gtime % 256;
        gtime++;
        // set valid bit
        fs->volume[FCB_base + 31] = 1;
        // printf("(open_write) fp = %d, start index = %d\n", i, ret_index);
        return i;
      }
    }
    // should never reach here
    assert(found_valid_FCB);
  }
}

/* Implement read operation */
/* fp is the FCB index */
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{

  u16 FCB_base = fs->SUPERBLOCK_SIZE + fp * (fs->FCB_SIZE);
  // invalid fp
  if ((fp >= fs->MAX_FILE_NUM) || (fs->volume[FCB_base + 31] == 0)) {
    printf("invalid fp\n");
    return;
  }

  u32 file_size = *((u32 *)&(fs->volume[FCB_base + 20]));
  u16 start_index = *((u16 *)&(fs->volume[FCB_base + 24])); // fetch block index in FCB
  u32 storge_start = fs->FILE_BASE_ADDRESS + start_index * fs->STORAGE_BLOCK_SIZE;
  // only read within this file's range
  if (size > file_size) {
    printf("read size is larger than the file size\n");
    size = file_size;
  }
  // copy value to output
  for (int i = 0; i < size; i++) {
    output[i] = fs->volume[storge_start + i];
  }
}

/* Implement write operation here */
__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp)
{
  u16 FCB_base = fs->SUPERBLOCK_SIZE + fp * (fs->FCB_SIZE);
  // invalid fp
  if ((fp >= fs->MAX_FILE_NUM) || (fs->volume[FCB_base + 31] == 0)) {
    printf("invalid fp\n");
    return fp;
  }

  u32 file_size = *((u32 *)&(fs->volume[FCB_base + 20]));
  u16 file_len;
  if (file_size == 0) {
    file_len = 1;
  }
  else {
    file_len = (file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE; // covered blocks
  }

  u32 write_len = (size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE; // covered blocks
  u16 start_index = *((u16 *)&fs->volume[FCB_base + 24]);                       // fetch block index in FCB
  u32 storge_start = fs->FILE_BASE_ADDRESS + start_index * fs->STORAGE_BLOCK_SIZE;

  u32 next_start = storge_start + file_len * fs->STORAGE_BLOCK_SIZE;
  u32 gap = file_len * fs->STORAGE_BLOCK_SIZE;
  u16 next_index = start_index + file_len;
  u16 end_block = get_0_bit_in_VCB(fs, 0); // get the first 0 bit in VCB
  u32 storge_end = fs->FILE_BASE_ADDRESS + end_block * fs->STORAGE_BLOCK_SIZE;

  // erase content
  for (int i = 0; i < file_size; i++) 
  {
    fs->volume[storge_start + i] = 0;
  }

  // write on the same location
  // VCB (bitmap) doesn't change
  if (write_len == file_len) {
    // printf("equal len, fp = %d, file_len = %d, start_idx = %d\n", fp, file_len, start_index);
    for (int i = 0; i < size; i++) 
    {
      fs->volume[storge_start + i] = input[i];
    }
    *((u32 *)&(fs->volume[FCB_base + 20])) = size;
    fs->volume[FCB_base + 28] = gtime / 256; // modified time
    fs->volume[FCB_base + 29] = gtime % 256;
    gtime++;
    return fp;
  }

  // write length != file length
  else
  {
    u32 empty_blocks = count_zeros(fs) + file_len;
    if (write_len > empty_blocks) 
    {
      printf("Exceed Maximum Storage.\n");
      return fp;
    }
    
    else
    {
      // last file, just keep writing
      if (next_start == storge_end)
      {
        for (int i = 0; i < size; i++)
        {
          fs->volume[storge_start + i] = input[i];
        }
        set_file_bits(fs, start_index, file_len, 0);
        set_file_bits(fs, start_index, write_len, 1);
        int tmp = get_0_bit_in_VCB(fs, 0);
        *((u32 *)&(fs->volume[FCB_base + 20])) = size;
        fs->volume[FCB_base + 28] = gtime / 256; // modified time
        fs->volume[FCB_base + 29] = gtime % 256;
        gtime++;
        // printf("\nlast file, fp = %d, write len = %d, start_idx = %d, next 0 index = %d\n", fp, write_len, start_index, tmp);
        return fp;
      }
      assert(next_index != end_block);
      // -------------- COMPACTION --------------
      // move the below contents upwards
      for (int i = next_start; i < storge_end; i++)
      {
        fs->volume[i - gap] = fs->volume[i];
      }

      // change FCB, should not involve this fp
      for (int i = 0; i < fs->FCB_ENTRIES; i++)
      {
        u32 fcb_base_ = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
        if (fs->volume[fcb_base_ + 31] != 0)
        {
          u16 fb_idx = *((u16 *)&(fs->volume[fcb_base_ + 24])); // fetch block index in FCB
          if (fb_idx >= next_index)                             // in the moved batch
          {
            *((u16 *)&(fs->volume[fcb_base_ + 24])) = fb_idx - file_len; // moving offset
          }
        }
      }

      // change VCB
      if (write_len > file_len)
      {
        set_file_bits(fs, end_block, (write_len - file_len), 1); // add bits
      }
      else if (write_len < file_len)
      {
        set_file_bits(fs, (end_block - file_len + write_len), (file_len - write_len), 0); // erase bits
      }


      u16 write_idx = end_block - file_len;
      u32 storge_write_start = fs->FILE_BASE_ADDRESS + write_idx * fs->STORAGE_BLOCK_SIZE;

      // clear the remainings
      for (int i = 0; i < gap; i++)
      {
        fs->volume[storge_write_start + i] = 0;
      }

      // write value
      for (int i = 0; i < size; i++)
      {
        fs->volume[storge_write_start + i] = input[i];
      }

      // change FCB for this fp
      *((u32 *)&(fs->volume[FCB_base + 20])) = size;
      *((u16 *)&(fs->volume[FCB_base + 24])) = write_idx;
      fs->volume[FCB_base + 28] = gtime / 256; // modified time
      fs->volume[FCB_base + 29] = gtime % 256;
      gtime++;

      // printf("compaction, fp = %d, file len = %d, start_idx = %d\n", fp, file_len, write_start);
      return fp;
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op)
{

  u16 f_idx[1024];
  for (int i = 0; i < fs->MAX_FILE_NUM; i++)
  {
    f_idx[i] = i;
  }

  if (op == LS_D)
  {
    int f_time[1024];

    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if (fs->volume[FCB_base + 31] != 0)
      {
        f_time[i] = (fs->volume[FCB_base + 28] * 256) + fs->volume[FCB_base + 29];
      }
      else
      {
        f_time[i] = -1;
      }
    }

    printf("===sort by modified time===\n");
    // insertion sort
    for (int i = 1; i < 1024; i++)
    {
      int key = f_time[i];
      u16 idx = f_idx[i];
      int j = i - 1;

      while (j >= 0 && key > f_time[j])
      {
        f_time[j + 1] = f_time[j];
        f_idx[j + 1] = f_idx[j];
        j = j - 1;
      }

      f_time[j + 1] = key;
      f_idx[j + 1] = idx;
    }

    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      if (f_time[i] == -1)
      {
        break;
      }
      else
      {
        char f_name[20];
        u16 idx = f_idx[i];
        u32 FCB_base = fs->SUPERBLOCK_SIZE + idx * fs->FCB_SIZE;
        _strcpy(f_name, (char *)&fs->volume[FCB_base]);
        printf("%s\n", f_name);
      }
    }
  }

  else if (op == LS_S)
  {
    int f_sizes[1024];
    int f_ctime[1024];

    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if (fs->volume[FCB_base + 31] != 0)
      {
        f_sizes[i] = *((u32 *)&(fs->volume[FCB_base + 20]));
        f_ctime[i] = (fs->volume[FCB_base + 26] * 256) + fs->volume[FCB_base + 27];
      }
      else
      {
        f_sizes[i] = -1;
        f_ctime[i] = -1;
      }
    }

    printf("===sort by file size===\n");

    // insertion sort
    for (int i = 1; i < 1024; i++)
    {
      int key = f_sizes[i];
      u16 idx = f_idx[i];
      int ct = f_ctime[i];

      int j = i - 1;
      while (j >= 0 && (key > f_sizes[j] || ((key == f_sizes[j]) && (ct < f_ctime[j]))))
      {
        f_sizes[j + 1] = f_sizes[j];
        f_idx[j + 1] = f_idx[j];
        f_ctime[j + 1] = f_ctime[j];

        j = j - 1;
      }
      f_sizes[j + 1] = key;
      f_ctime[j + 1] = ct;
      f_idx[j + 1] = idx;
    }

    for (int i = 0; i < 1024; i++)
    {
      if (f_sizes[i] == -1)
      {
        break;
      }
      else
      {
        char f_name[20];
        u16 idx = f_idx[i];
        u32 FCB_base = fs->SUPERBLOCK_SIZE + idx * fs->FCB_SIZE;
        _strcpy(f_name, (char *)&fs->volume[FCB_base]);
        printf("%s %d\n", f_name, f_sizes[i]);
      }
    }
  }
}

/* Implement rm operation here */
__device__ void fs_gsys(FileSystem *fs, int op, char *filename)
{
  if (op == RM)
  {
    for (int i = 0; i < fs->FCB_ENTRIES; i++)
    {
      u32 FCB_base = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
      if (_strcmp(filename, (char *)&(fs->volume[FCB_base])) == 0)
      {

        u32 file_size = *((u32 *)&fs->volume[FCB_base + 20]);
        u16 start_index = *((u16 *)&fs->volume[FCB_base + 24]);
        u16 file_len;

        if (file_size == 0)
        {
          file_len = 1;
        }
        else
        {
          file_len = (file_size + fs->STORAGE_BLOCK_SIZE - 1) / fs->STORAGE_BLOCK_SIZE;
        }

        set_file_bits(fs, start_index, file_len, 0); // remove bits

        // clear FCB
        for (int j = 0; j < fs->FCB_SIZE; j++)
        {
          fs->volume[FCB_base + j] = 0;
        }

        return;
      }
    }
    // should never reach here
    printf("Cannot find the matched file!\n");
    return;
  }
}