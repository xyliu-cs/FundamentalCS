#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>
#include <stdio.h>


typedef struct Task {
  struct Task *next;
  struct Task *prev;
  void (*target_function) (int);
  int arg;
} Task;


typedef struct task_queue {
  int size;
  Task *head;
} task_queue;

void async_init(int);
void async_run(void (*fx)(int), int args);
void *starter(void *);
#endif
