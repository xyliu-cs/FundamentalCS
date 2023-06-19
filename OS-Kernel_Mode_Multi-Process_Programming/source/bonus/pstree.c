#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <ctype.h>
#include <getopt.h>

struct proc_node {
	char name[200];
	pid_t pid;
	pid_t ppid;
	int child_count;
	struct proc_node *parent;
	struct proc_node *children[500];
	struct proc_node *next;
};

// the head of the structed linked list
static struct proc_node *head_ptr = NULL;
char *pids[2000];

char *trimString(char *str)
{
	char *end;

	while (isspace((unsigned char)*str))
		str++;

	if (*str == 0)
		return str;

	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end))
		end--;

	end[1] = '\0';

	return str;
}

char *get_attribute(char *buff)
{
	char exp[200];
	char *token;
	char *ret;

	const char s[] = ":";

	strcpy(exp, buff);
	token = strtok(exp, s);
	token = strtok(NULL, s); // second part
	ret = trimString(token);
	// printf("%s\n", ret);

	return ret;
}

int create_insert_node(char *name, pid_t pid, pid_t ppid)
{
	struct proc_node *ptr;
	ptr = (struct proc_node *)malloc(sizeof(struct proc_node));
	if (ptr == NULL) {
		printf("Unable to allocate memory for node\n");
		return 1;
	}

	strcpy(ptr->name, name);
	ptr->pid = pid;
	ptr->ppid = ppid;
	ptr->child_count = 0;
	ptr->children[0] = NULL;
	ptr->parent = NULL;
	ptr->next = head_ptr;
	head_ptr = ptr;

	return 0;
}

int struct_node_attr(char *pid, char *p_path)
{
	char path[300];
	FILE *fp;
	char strbuff[300];
	const char name[] = "Name:";
	const char ppid[] = "PPid";
	const char tgid[] = "Tgid";

	// construct this tree node
	struct proc_node this_node;

	char proc_name[200];
	char proc_pid[32];
	char proc_ppid[32];
	char proc_tgid[32];

	char td_name[200];

	strcpy(proc_pid, pid);

	strcpy(path, p_path);
	strcat(path, "/status");
	fp = fopen(path, "r");
	if (fp == NULL) {
		// printf("empty path %s\n", path);
		return 0;
	}

	// printf("My pid is : %s\n", pid);

	while (fgets(strbuff, 300, fp) != NULL) {
		if (strstr(strbuff, name) != NULL) {
			strcpy(proc_name, get_attribute(strbuff));
			// printf("My name is: %s\n", proc_name);

		} else if (strstr(strbuff, ppid) != NULL) {
			strcpy(proc_ppid, get_attribute(strbuff));
			// printf("My ppid is: %s\n", proc_ppid);

		} else if (strstr(strbuff, tgid) != NULL) {
			strcpy(proc_tgid, get_attribute(strbuff));
			// printf("My tgid is: %s\n", proc_ppid);
		}
	}
	// printf("My pid is: %s\n", proc_pid);

	int tmp_pid = atoi(proc_pid);
	int tmp_ppid = atoi(proc_ppid);
	int tmp_tgid = atoi(proc_tgid);

	if (tmp_tgid != tmp_pid) {
		strcpy(td_name, "{");
		strcat(td_name, proc_name);
		strcat(td_name, "}");
		create_insert_node(td_name, tmp_pid, tmp_tgid);
		return 0;
	}

	// printf("=========================\n");
	create_insert_node(proc_name, tmp_pid, tmp_ppid);
	// printf("inserted\n");
	// printf("=========================\n");

	return 0;
}

struct proc_node *find_node(pid_t pid)
{
	struct proc_node *ptr;
	struct proc_node *null_ptr = NULL;

	for (ptr = head_ptr; ptr != NULL; ptr = ptr->next) {
		if (ptr->pid == pid) {
			return ptr;
		}
	}
	printf("Can't find this pid %d\n", pid);
	return 0;
}

void parenting(pid_t child_pid, pid_t parent_pid)
{
	struct proc_node *chld_ptr;
	struct proc_node *prnt_ptr;
	int tmp;

	//first process
	if (parent_pid == 0) {
		return;
	}

	// printf("Child pid is %d\n", chld_ptr->pid );
	// printf("Parent pid is %d\n", prnt_ptr->pid );

	// both found
	if (find_node(child_pid) && find_node(parent_pid)) {
		chld_ptr = find_node(child_pid);
		prnt_ptr = find_node(parent_pid);
		chld_ptr->parent = prnt_ptr;
		tmp = prnt_ptr->child_count;
		prnt_ptr->children[tmp] = chld_ptr;
		prnt_ptr->child_count += 1;
		// printf("parenting success\n");
	} else {
		printf("Can't match. Child pid: %d, Parent pid: %d\n",
		       child_pid, parent_pid);
	}
};

void form_tree(void)
{
	struct proc_node *local_ptr;
	struct proc_node *prnt_ptr;
	int tmp;
	// printf("start HERE ----------------\n");
	int i = 0;
	for (local_ptr = head_ptr; local_ptr != NULL;
	     local_ptr = local_ptr->next) {
		// printf("count: %d\n", i);
		// i++;
		// if(i >= 1000) break;
		// printf("node: %s, pid: %d, ppid: %d\n", local_ptr->name, local_ptr->pid, local_ptr->ppid);
		if (local_ptr->ppid == 0) {
			local_ptr->parent == NULL;
			continue;
		}
		// local_ptr->parent = find_node(local_ptr->ppid);
		// prnt_ptr = local_ptr->parent;
		// tmp = prnt_ptr->child_count;
		// prnt_ptr->children[tmp] = local_ptr;
		// prnt_ptr->child_count += 1;

		parenting(local_ptr->pid, local_ptr->ppid);
	}
	// printf("success\n");
}

int print_tree(struct proc_node *root, int level, int flag)
{
	int i;
	struct proc_node *node;
	for (i = 0; i < level; i++) {
		printf("    ");
	}
	// printf("%s\n", root->name);

	printf("|-");
	switch (flag) {
	case 0:
		printf("%s\n", root->name);
		break;
	case 1:
		printf("%s (%d)\n", root->name, root->pid);
	}

	/* recurse on children */
	int j = 0;
	while ((node = root->children[j++]) != NULL) {
		print_tree(node, level + 1, flag);
	}
	return 0;
}

int main(int argc, char *argv[])
// int main ()
{
	struct dirent *dirp;
	struct dirent *dirp_task;
	char pid_path[100];
	char pid_path_task[150];
	char pid_path_task_new[150];

	char pid[20];
	char pid_task[20];
	char buff_dname[256];

	// command line args
	extern char *optarg;
	extern int optind, opterr, optopt;
	char opt;

	// arg type
	int flag = 0;

	DIR *dir = opendir("/proc");
	DIR *dir_task;

	if (dir == NULL) {
		printf("failure to open /proc directory!\n");
	}
	while (1) {
		dirp = readdir(dir);

		// readdir: open directory
		if (dirp == NULL) {
			break;
		}
		// dirp->d_type 是这个指针指向文件的类型
		// DT_DIR  目录
		// DT_REG  文件

		if (dirp->d_type == DT_DIR) {
			strcpy(pid_path, "/proc/");
			strcpy(pid, dirp->d_name);
			strcat(pid_path, dirp->d_name);
			// pid_path is a pointer to head of the array
			// printf("the path1: %s\n", pid_path);
			struct_node_attr(pid, pid_path);

			strcpy(pid_path_task, pid_path);
			strcat(pid_path_task, "/task");
			dir_task = opendir(pid_path_task);
			// skip empty dir
			if (dir_task != NULL) {
				while (1) {
					dirp_task = readdir(dir_task);
					if (dirp_task == NULL) {
						break;
					}
					// threads under proc/<pid>/task
					if (dirp_task->d_type == DT_DIR) {
						// skip itself
						strcpy(buff_dname,
						       dirp_task->d_name);
						if ((strcmp(buff_dname, pid) !=
						     0) &&
						    (buff_dname[0] != '.')) {
							/* code */
							strcpy(pid_path_task_new,
							       "/proc/");
							strcat(pid_path_task_new,
							       dirp_task->d_name);
							strcpy(pid_task,
							       dirp_task->d_name);
							struct_node_attr(
								pid_task,
								pid_path_task_new);
							// printf("the path2: %s\n", pid_path_task_new);
						}
					}
				}
			}

			// printf("the path: %s\n", pid_path);

			// printf("SUCCESS HERE ----------------");
		}
	}

	/* code */

	form_tree();

	while ((opt = getopt(argc, argv, "Apl")) != -1) {
		switch (opt) {
		case 'A':
			flag = 0;
			break;
		case 'p': //若解析到的返回值为n，即选项-n且加了参数
			flag = 1; //将选项-n的参数字符串转为整数
			break;
		case 'l':
			flag = 0;
			break;
		case '?':
			flag = 0;
			break;
		}
	}

	/* print the tree */
	struct proc_node *node;
	printf("---------------- PRINT START ----------------\n");
	for (node = head_ptr; node != NULL; node = node->next) {
		//     printf("node: %s, pid: %d, ppid: %d\n", node->name, node->pid, node->ppid);
		if (node->parent == NULL) {
			print_tree(node, 0, flag);
		}
	}

	closedir(dir);
	printf("---------------- PRINT FINISHED ----------------\n");

	return 0;
}
