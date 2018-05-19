#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <utils.h>
#include <cmdl.h>
#include <emmintrin.h>
#include "dSFMT.h"
#define SQR(x) ((x)*(x))
#define max(a,b) ((a)>(b)?(a):(b)) 
dsfmt_t dsfmt;
typedef struct particle{
	unsigned flag,n;
	__m128d *r; //particle center of mass;
}particle;
typedef struct vertex{
	__m128d *r;
}vertex;
typedef struct edge{
	int j[2];
	vertex *v[2];
	void *t[2];
}edge;
typedef struct triangle{
	int flag;
	int depth;
	int dir[3];
	__m128d c;
	edge *e[3];
	struct triangle *t[3];
	struct triangle *t0;
	struct triangle *left,*right,*next;
}triangle;
typedef struct obj{
	triangle *t;
	edge *e;
	vertex *v;
	int nvertex;
	int nedge;
	int ntriangle;
}obj;
typedef struct node{
	int depth;
	int h;
	vertex *v;
	double key;
	struct node *left;
	struct node *right;
	struct node *next;
}node;
int dump_particle(particle *p,unsigned int n){
	unsigned int i;
	for(i=0;i<n;i++){
		printf("%lf %lf\n",(*(p+i)->r)[0],(*(p+i)->r)[1]);
	}
	return 0;
}
int dump_m128d(__m128d r){
		printf("%lf %lf\n",r[0],r[1]);
	return 0;
}
int dump_vertex(vertex *v){
		printf("%lf %lf\n",(*(v)->r)[0],(*(v)->r)[1]);
	return 0;
}
int dump_all_vertex(vertex *v,unsigned int n){
	unsigned int i;
	for(i=0;i<n;i++){
		printf("%lf %lf\n",(*(v+i)->r)[0],(*(v+i)->r)[1]);
	}
	return 0;
}
int dump_edge(edge *e,int dir){
		printf("%lf %lf ",(*(e->v[dir])->r)[0],(*(e->v[dir])->r)[1]);
		printf("%lf %lf\n",(*(e->v[1^dir])->r)[0]-(*(e->v[dir])->r)[0],(*((e->v[1^dir]))->r)[1]-(*(e->v[dir])->r)[1]);
	return 0;
}
int dump_edge2(edge *e,int dir){
		printf("%d %lf %lf ",e->j[dir],(*(e->v[dir])->r)[0],(*(e->v[dir])->r)[1]);
		printf("%lf %lf\n",(*(e->v[1^dir])->r)[0]-(*(e->v[dir])->r)[0],(*((e->v[1^dir]))->r)[1]-(*(e->v[dir])->r)[1]);
	return 0;
}
int dump_triangle(triangle *t){
	dump_edge(*(t->e),t->dir[0]);
	dump_edge(*(t->e+1),t->dir[1]);
	dump_edge(*(t->e+2),t->dir[2]);
	return 0;
}
int dump_all_edge(edge *e,unsigned int n){
	unsigned int i;
	for(i=0;i<n;i++){
		dump_triangle(((e+i)->t)[0]);
		printf("\n");
	}
	return 0;
}
particle *particle_alloc(unsigned int n){
	unsigned int i;
	particle *p=(particle*)alloc(sizeof(particle)*n);
	p->r=(__m128d*)alloc(n*sizeof(__m128d));
	for(i=0;i<n;i++){
		(p+i)->r=(p->r)+i;
	}
	return p;
}
int init_particle(particle *p,unsigned int n,__m128d box){
	unsigned int i;
	double a,b;
	__m128d t;
	for(i=0;i<n;i++){
		a=dsfmt_genrand_open_open(&dsfmt);
		b=dsfmt_genrand_open_open(&dsfmt);
		t=box*(__m128d){a,b};
		*(p+i)->r=t;
	}
	return 0;
}
vertex *vertex_alloc(particle *p,unsigned int n){
	unsigned int i;
	vertex *v=(vertex*)alloc(sizeof(vertex)*n);
	for(i=0;i<n;i++){
		(v+i)->r=(p+i)->r;
	}
	return v;
}
edge *edge_alloc(unsigned int n){
	edge *e=(edge*)alloc(sizeof(edge)*n);
	return e;
}
triangle *triangle_alloc(unsigned int n){
	triangle *t=(triangle*)alloc(sizeof(triangle)*n);
	unsigned int i;
	for(i=0;i<n;i++){
		(t+i)->t0=t;
		(t+i)->flag=0;
		(t+i)->depth=0;
		(t+i)->left=NULL;
		(t+i)->right=NULL;
		(t+i)->next=NULL;
	}
	return t;
}
int edge_assign_vertex(edge *e,vertex *v1,vertex *v2){
	*(e->v)=v1;
	*((e->v)+1)=v2;
	return 0;
}
int triangle_assign_edge(triangle *t,edge *e0,edge *e1,edge *e2,int d0,int d1,int d2){
	*(t->e)=e0;
	*(t->dir)=d0;
	e0->t[d0]=t;
	e0->j[d0]=0;

	*((t->e)+1)=e1;
	*((t->dir)+1)=d1;
	e1->t[d1]=t;
	e1->j[d1]=1;

	*((t->e)+2)=e2;
	*((t->dir)+2)=d2;
	e2->t[d2]=t;
	e2->j[d2]=2;
	return 0;
}
int triangle_assign_edge2(triangle *t,triangle *p,edge *e0,edge *e1,edge *e2,int d0,int d1,int d2){
	*(t->e)=e0;
	*(t->dir)=d0;
	e0->t[d0]=p;
	e0->j[d0]=0;

	*((t->e)+1)=e1;
	*((t->dir)+1)=d1;
	e1->t[d1]=p;
	e1->j[d1]=1;

	*((t->e)+2)=e2;
	*((t->dir)+2)=d2;
	e2->t[d2]=p;
	e2->j[d2]=2;
	return 0;
}
double cross(__m128d a,__m128d b){
	return a[0]*b[1]-a[1]*b[0];
}
vertex *get_vertex(triangle *t,int edge){
	return t->e[edge]->v[t->dir[edge]];
}
__m128d get_m128d(triangle *t,int edge){
	return *t->e[edge]->v[t->dir[edge]]->r;
}
int intersect(edge *e1,edge *e2){//returns 1 at intersecting
	double rs,t,u;
	__m128d p=*e1->v[0]->r,q=*e2->v[0]->r;
	__m128d r=*e1->v[1]->r-p,s=*e2->v[1]->r-q;
	__m128d qp;
	qp=q-p;
	rs=cross(r,s);
	if(rs==0.0){
		return 0;
	}
	rs=1.0/rs;
	t=cross(qp,s)*rs;
	u=cross(qp,r)*rs;
	if((0.0>t)||(t>1.0)||(0.0>u)||(u>1.0)){
		return 0;
	}
	return 1;
}
int intersect_triangle_edge(triangle *t,edge *e){
	unsigned int i;
	edge *f;
	for(i=0;i<3;i++){
		f=t->e[i];	
		if(intersect(e,f)){
			return i;
		}
	}
	return 3;
}
int intersect_triangle_edge2(triangle *t,edge *e,edge *last){
	unsigned int i;
	edge *f;
	for(i=0;i<3;i++){
		f=t->e[i];	
		if(f!=last&&intersect(e,f)){
				return i;
		}
	}
	return 3;
}
__m128d triangle_t(triangle *t){
	__m128d c=get_m128d(t,0)+get_m128d(t,1)+get_m128d(t,2);
	return c/(__m128d){3.0,3.0};
}
int triangle_c(triangle *t){
	t->c=(__m128d)(get_m128d(t,0)+(__m128d)get_m128d(t,1)+(__m128d)get_m128d(t,2))/(__m128d){3.0,3.0};
	return 0;
}
int dump_triangle_walk(triangle *t,edge *e){
	int j,k=1;
	dump_triangle(t);
	putchar('\n');
	*e->v[1]->r=triangle_t(t);
	j=intersect_triangle_edge(t,e);
	if(j==3){
		return 0;
	}
	t=t->e[j]->t[1^t->dir[j]];	
	dump_edge(e,1);
	putchar('\n');
	dump_triangle(t);
	putchar('\n');
	while(j!=3){
		*e->v[1]->r=triangle_t(t);
		j=intersect_triangle_edge(t,e);
		if(j==3)break;
		t=t->e[j]->t[1^t->dir[j]];	
		dump_edge(e,1);
		putchar('\n');
		dump_triangle(t);
		putchar('\n');
		k++;
	}
	printf("k=%d\n",k);
	return 0;
}
int dump_triangle_walk2(triangle *t,edge *e){
	int j,k=1;
	edge *last;
	dump_triangle(t);
	putchar('\n');
	*e->v[1]->r=triangle_t(t);
	j=intersect_triangle_edge(t,e);
	if(j==3){
		return 0;
	}
	last=t->e[j];
	t=t->e[j]->t[1^t->dir[j]];	
	dump_edge(e,1);
	putchar('\n');
	dump_triangle(t);
	putchar('\n');
	while(j!=3){
		j=intersect_triangle_edge2(t,e,last);
		if(j==3)break;
		last=t->e[j];
		t=t->e[j]->t[1^t->dir[j]];	
		dump_edge(e,1);
		putchar('\n');
		dump_triangle(t);
		putchar('\n');
		k++;
	}
	printf("k=%d\n",k);
	return 0;
}
triangle *triangle_walk(triangle *t,edge *e){
	int j;
	int k=0;
	edge *last;
	*e->v[1]->r=triangle_t(t);
	j=intersect_triangle_edge(t,e);
	if(j==3){
		printf("%d\n",k);
		return t;
	}
	last=t->e[j];
	t=t->e[j]->t[1^t->dir[j]];	
	while(j!=3){
		k++;
		j=intersect_triangle_edge2(t,e,last);
		if(j==3){
			printf("%d\n",k);
			return t;
		}
		last=t->e[j];
		t=t->e[j]->t[1^t->dir[j]];	
	}
	return NULL;
}
int point_in_triangle(triangle *t,vertex *s){
	double a,b,ic;
	__m128d p=*(s->r);
	__m128d v3=*(get_vertex(t,2)->r);
	__m128d v13=*(get_vertex(t,0)->r)-v3;
	__m128d v23=*(get_vertex(t,1)->r)-v3;

	ic=1.0/cross(v13,v23);
	a=(cross(p,v23)-cross(v3,v23))*ic;
	b=-(cross(p,v13)-cross(v3,v13))*ic;

	if((a>=0.0)&&(b>=0.0)&&((a+b)<=1.0))return 0;
	else return 1;
}
int point_in_circumcircle(edge *e){
	double d;
	vertex *v;
	triangle *t=e->t[0];
	triangle *t1=e->t[1];
	int j=(e->j[1]+2)%3;
	if(t==NULL||t1==NULL){
		return 1;
	}
	v=get_vertex(t1,j);

	__m128d p=*(v->r);
	__m128d a=*(get_vertex(t,0)->r);
	__m128d b=*(get_vertex(t,1)->r);
	__m128d c=*(get_vertex(t,2)->r);

	__m128d ap=a-p;
	__m128d bp=b-p;
	__m128d cp=c-p;

	__m128d p2=p*p;
	__m128d a2=a*a;
	__m128d b2=b*b;
	__m128d c2=c*c;

	__m128d ap2=a2-p2;
	__m128d bp2=b2-p2;
	__m128d cp2=c2-p2;

	double aa=ap2[0]+ap2[1];
	double bb=bp2[0]+bp2[1];
	double cc=cp2[0]+cp2[1];

	d=ap[0]*bp[1]*cc+ap[1]*bb*cp[0]+aa*bp[0]*cp[1]-aa*bp[1]*cp[0]-ap[1]*bp[0]*cc-ap[0]*bb*cp[1];
	if(d>0.0)return 0;
	else return 1;
}
int flip_edge(edge *e){
	triangle *pt1=e->t[0],*pt2=e->t[1],t2=*pt2;
	int i1=e->j[0],i2=e->j[1],i11=(i1+1)%3,i12=(i1+2)%3,i21=(i2+1)%3,i22=(i2+2)%3;

	edge_assign_vertex(e,pt2->e[i22]->v[pt2->dir[i22]],pt1->e[i12]->v[pt1->dir[i12]]); //Flip edge

	triangle_assign_edge2(&t2,pt2,pt2->e[i22],pt1->e[i11],e,pt2->dir[i22],pt1->dir[i11],1);
	triangle_assign_edge(pt1,pt1->e[i12],pt2->e[i21],e,pt1->dir[i12],pt2->dir[i21],0);
	*pt2=t2;
	return 0;
}
int flip_all_edge(edge *e,unsigned int ne){
	unsigned int i;
	for(i=0;i<ne;i++){
		if(!point_in_circumcircle(e+i)){;
			flip_edge(e+i);
		}
	}
	return 0;
}
int triangle_flip(triangle *t){
	unsigned int i;
	edge *e;
	triangle *tt;
	for(i=0;i<3;i++){
		e=t->e[i];
		if(!point_in_circumcircle(e)){;
			flip_edge(e);
			tt=e->t[0];
			tt->flag=2;
			tt=e->t[1];
			tt->flag=2;
			triangle_flip(e->t[0]);
			triangle_flip(e->t[1]);
		}
	}
	return 0;
}
int tesselate_point(triangle *t,edge *e,vertex *s,unsigned int *nt,unsigned *ne){
	triangle *t0=t->t0+*nt;
	edge *e0=e+*ne;

	edge_assign_vertex(e0,t->e[1]->v[t->dir[1]],s);
	edge_assign_vertex(e0+1,s,t->e[0]->v[t->dir[0]]);
	edge_assign_vertex(e0+2,t->e[2]->v[t->dir[2]],s);

	triangle_assign_edge(t0,t->e[0],e0,e0+1,t->dir[0],0,0);
	triangle_assign_edge(t0+1,t->e[1],e0+2,e0,t->dir[1],0,1);
	triangle_assign_edge(t0+2,t->e[2],e0+1,e0+2,t->dir[2],1,1);

	triangle_c(t0);
	triangle_c(t0+1);
	triangle_c(t0+2);

	(*nt)+=3;
	(*ne)+=3;
	t->flag=1;

	triangle_flip(t);
	return 0;
}
int tesselate(triangle *t,edge *e,vertex *v,unsigned int *nt,unsigned int *ne,unsigned int nv){
	unsigned int i,n=*nt;
	for(i=4;i<nv;i++){
		n=*nt;
		for(unsigned j=0;j<n;j++){
			if((t+j)->flag!=1){
				if(!point_in_triangle(t+j,v+i)){
					//printf("j=%d\n",j);
					tesselate_point(t+j,e,v+i,nt,ne);
					break;
				}
			}
		}
	}
	return 0;
}
int insert_kd_triangle(triangle **root,triangle *t,int *depth){
	int b=*depth%2;
	double c=t->c[b],k;
	if(*root==NULL){
		*root=t;
		t->depth=*depth;
	}
	else{
		k=((*root)->c)[b];
		if(c<k){
			(*depth)++;
			insert_kd_triangle(&(*root)->left,t,depth);
		}
		else{
			(*depth)++;
			insert_kd_triangle(&(*root)->right,t,depth);
		}
	}
	return 0;
}
triangle *close_kd_triangle(triangle* root,vertex *v){
	__m128d d;
	double m=128.0,mm;
	int  b;
	int n=0;
	triangle *res=NULL;
	triangle *current;
	__m128d x=*v->r;
	current=root;
	while(current!=NULL){
		d=current->c-x;
		mm=fabs(d[0])+fabs(d[1]);
		if(current->flag!=1&&mm<m){
			m=mm;
			res=current;
		}
		b=current->depth%2;
		if(current->c[b]>(x)[b]){
			current=current->left; 
			n++;
		}
		else{
			current=current->right;
			n++;
		}
	}
	return res;
}
void nearest(triangle *root,vertex *nd,int i,triangle **best,double *best_dist){
	double d, dx, dx2;
	__m128d d2;
	if(!root)return;
	d=0;
	d2=root->c-*nd->r;
	d=SQR(d2[0])+SQR(d2[1]);
	dx=root->c[i]-(*nd->r)[i];
	dx2=dx*dx;
	if((root->flag!=1)&&(!*best||d<*best_dist)){
		*best_dist=d;
		*best=root;
	}
	if(++i>=2)i=0;
	nearest(dx>0?root->left:root->right,nd,i,best,best_dist);
		if(dx2>=*best_dist)return;
	nearest(dx>0?root->right:root->left,nd,i,best,best_dist);
}
int build_kd_triangle(triangle *t){
	int depth=0;
	insert_kd_triangle(&t,t+1,&depth);
	return 0;
}
triangle *rebuild_kd_triangle(triangle *t,unsigned int nt){
	int depth=0;
	unsigned int i=0,j;
	triangle *tt=t;
	while(tt->flag==0){
		t++;
		i++;
	}
	tt->left=NULL;
	tt->right=NULL;
	for(j=i;j<nt;j++){
		(t+j)->left=NULL;
		(t+j)->right=NULL;
		if((t+j)->flag==2){
			triangle_c(t+j);
			(t+j)->flag=0;
		}
		if((t+j)->flag!=1){
			depth=0;
			insert_kd_triangle(&tt,t+j,&depth);
		}
	}
	return tt;
}
int tesselate_walk_kd(triangle *t,edge *e,vertex *v,unsigned int *nt,unsigned int *ne,unsigned int nv){
	unsigned int i;
	int depth=0;
	double best_dist;
	triangle *p,*tt;
	edge ee;
	ee.v[1]=(vertex*)alloc(sizeof(vertex));
	ee.v[1]->r=(__m128d*)alloc(sizeof(__m128d));
	tt=t;
	for(i=4;i<nv;i++){
		depth=0;
		best_dist=100000;
		nearest(tt,v+i,depth,&p,&best_dist);
		if(i==10000||i==200000){
			tt=rebuild_kd_triangle(tt,*nt);
		}
		ee.v[0]=v+i;
		p=triangle_walk(p,&ee);
		tesselate_point(p,e,v+i,nt,ne);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-1,&depth);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-3,&depth);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-2,&depth);
	}
	return 0;
}
int tesselate_walk_kd2(triangle *t,edge *e,vertex *v,unsigned int *nt,unsigned int *ne,unsigned int nv){
	unsigned int i;
	int depth=0;
	triangle *p,*tt;
	edge ee;
	ee.v[1]=(vertex*)alloc(sizeof(vertex));
	ee.v[1]->r=(__m128d*)alloc(sizeof(__m128d));
	tt=t;
	for(i=4;i<nv;i++){
		p=close_kd_triangle(tt,v+i);
		if(!p){
			p=t+*nt-1;
		}
		ee.v[0]=v+i;
		p=triangle_walk(p,&ee);
		tesselate_point(p,e,v+i,nt,ne);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-1,&depth);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-2,&depth);
		depth=0;
		insert_kd_triangle(&tt,t+*nt-3,&depth);
	}
	return 0;
}
int tesselate_walk(triangle *t,edge *e,vertex *v,unsigned int *nt,unsigned int *ne,unsigned int nv){
	unsigned int i;
	int depth=0;
	triangle *p;
	edge ee;
	ee.v[1]=(vertex*)alloc(sizeof(vertex));
	ee.v[1]->r=(__m128d*)alloc(sizeof(__m128d));
	for(i=4;i<nv;i++){
		ee.v[0]=v+i;
		p=triangle_walk(t+*nt-1,&ee);
		tesselate_point(p,e,v+i,nt,ne);
		triangle_c(t+*nt-3);
		insert_kd_triangle(&t,t+*nt-3,&depth);
		triangle_c(t+*nt-2);
		insert_kd_triangle(&t,t+*nt-2,&depth);
		triangle_c(t+*nt-1);
		insert_kd_triangle(&t,t+*nt-1,&depth);
	}
	return 0;
}
int print_tree_triangle(triangle *root){
	if(root->left){
		print_tree_triangle(root->left);
	}
	if(root->flag!=1){
		dump_m128d(root->c);
	}
	if(root->right){
		print_tree_triangle(root->right);
	}
	return 0;
}
int init_square(triangle *t,edge *e,vertex *v,__m128d x){
	unsigned int i;
	*(v->r)=(__m128d){0.0,0.0};
	*((v+1)->r)=x;
	*((v+2)->r)=(__m128d){0.0,x[1]};
	*((v+3)->r)=(__m128d){x[0],0.0};

	edge_assign_vertex(e,v,v+1);
	edge_assign_vertex(e+1,v+1,v+2);
	edge_assign_vertex(e+2,v+2,v);
	edge_assign_vertex(e+3,v,v+3);
	edge_assign_vertex(e+4,v+3,v+1);

	triangle_assign_edge(t,e,e+1,e+2,0,0,0);
	triangle_assign_edge(t+1,e+3,e+4,e,0,0,1);

	triangle_c(t);
	triangle_c(t+1);

	for(i=1;i<5;i++){
		((e+i)->t)[1]=NULL;
	}
	return 0;
}
node *alloc_node(double key,vertex *v){
	node *a=(node*)alloc(sizeof(node));
	a->key=key;
	a->h=1;
	a->v=v;
	a->left=NULL;
	a->right=NULL;
	a->next=NULL;
	return a;
}
node *alloc_kd_node(double key,vertex *v,int depth){
	node *a=(node*)alloc(sizeof(node));
	a->key=key;
	a->h=1;
	a->depth=depth;
	a->v=v;
	a->left=NULL;
	a->right=NULL;
	a->next=NULL;
	return a;
}
int th(node *t){
	if(t==NULL)return 0;
	else return t->h;
}
int thm(node *t){
	return max(th(t->left),th(t->right))+1;
}
int tb(node *t){
	if(t==NULL)return 0;
	else return (th(t->left)-th(t->right));
}
node *rotate_right(node *root){
	if(root==NULL)return root;
	node *b=root->left;
	node *t=b->right;
	b->right=root;
	root->left=t;
	root->h=thm(root);
	b->h=thm(b);
	return b;
}
node *rotate_left(node *root){
	if(root==NULL)return root;
	node *b=root->right;
	node *t=b->left;
	b->left=root;
	root->right=t;
	root->h=thm(root);
	b->h=thm(b);
	return b;
}
int insert_node(node **root,double key,vertex *v){
	int b;
	if(*root==NULL){
		*root=alloc_node(key,v);
	}
	else if(key<(*root)->key){
		insert_node(&(*root)->left,key,v);
	}
	else{
		insert_node(&(*root)->right,key,v);
	}
	(*root)->h=thm(*root);
	b=tb(*root);
	if(b>1){
		if(key<(*root)->left->key){
			(*root)=rotate_right(*root);
		}
		else{
			(*root)->left=rotate_left((*root)->left);
			(*root)=rotate_right(*root);
		}
	}
	else if(b<-1){
		if(key>(*root)->right->key){
			(*root)=rotate_left(*root);
		}
		else{
			(*root)->right=rotate_right((*root)->right);
			(*root)=rotate_left(*root);
		}
	}
	return 0;
}
int insert_kd_node(node **root,double key,vertex *v,int *depth){
	int b=*depth%2;
	key=(*v->r)[b];
	if(*root==NULL){
		*root=alloc_kd_node(key,v,*depth);
	}
	else if(key<(*root)->key){
		(*depth)++;
		insert_kd_node(&(*root)->left,key,v,depth);
	}
	else{
		(*depth)++;
		insert_kd_node(&(*root)->right,key,v,depth);
	}
	return 0;
}
int print_tree(node *root,int d,int h){
	if(root->left){
		print_tree(root->left,0,h+1);
	}
	printf("rh:%d  d:%d h:%d\t",root->h,d,h);
	dump_vertex(root->v);
	if(root->right){
		print_tree(root->right,1,h+1);
	}
	return 0;
}
int print_tree_preorder(node *root,int d,int h){
	printf("rh:%d  d:%d h:%d\t",root->h,d,h);
	dump_vertex(root->v);
	if(root->left){
		print_tree(root->left,0,h+1);
	}
	if(root->right){
		print_tree(root->right,1,h+1);
	}
	return 0;
}
int print_tree_morris(node *root) {
	node *current,*pre;
	if(root==NULL){
		return 0; 
	}
	current=root;
	while(current!=NULL){                 
		if(current->left==NULL){
			dump_vertex(current->v);
			current=current->right;      
		}    
		else{
			pre = current->left;
			while(pre->right!=NULL&&pre->right!=current)
			pre=pre->right;
			if(pre->right==NULL){
				pre->right=current;
				current=current->left;
			}
			else{
				pre->right=NULL;
				dump_vertex(current->v);
				current=current->right;      
			}
		}
	}
	return 0;
}
void push_queue(node *t,node **q){
	if(t->left){
		(*q)->next=t->left;
		(*q)=(*q)->next;
		if(t->right){
			(*q)->next=t->right;
			(*q)=(*q)->next;
		}
	}
	else if(t->right){
		(*q)->next=t->right;
		(*q)=(*q)->next;
	}
}
int print_tree_level(node *root){
	node *queue,*last=root;
	dump_vertex(root->v);
	push_queue(root,&last);
	queue=root->next;
	while(queue!=NULL){
		dump_vertex(queue->v);
		push_queue(queue,&last);
		queue=queue->next;
	}
	return 0;
}
node *close(node* root,double key){
	double d,m=128.0;
	node *res=NULL;
	node *current=root;
	while(current!=NULL){
		d=fabs(current->key-key);
		if(d<m){
			m=d;
			res=current;
		}
		if(d==0.0){
			break;
		}
		if(current->key>key){
			current=current->left; 
		}
		else if(current->key<key){
			current=current->right;
		}
	}
	return res;
}
node *close_kd(node* root,vertex *v){
	double d[2],m[2]={128.0,128.0};
	double key;
	int  b;
	int n=0;
	node *res=NULL;
	node *current=root;
	while(current!=NULL){
		b=current->depth%2;
		key=(*v->r)[0];
		d[0]=fabs((*current->v->r)[0]-key);
		key=(*v->r)[1];
		d[1]=fabs((*current->v->r)[1]-key);
		//printf("n=%d\n",n);
		dump_vertex(current->v);
		if(d[0]+d[1]<m[0]){
			m[0]=d[0]+d[1];
			res=current;
			//printf("d[0]+d[1] %lf\n",d[0]+d[1]);
			dump_vertex(res->v);
		}
		key=(*v->r)[b];
		if(current->key>key){
			current=current->left; 
			n++;
		}
		//else if(current->key<key){
		else{
			current=current->right;
			n++;
		}
	}
	printf("\nn=%d\n",n);
	return res;
}
node *build_tree(vertex *v,unsigned int n){
	unsigned int i;
	double x=(*(v)->r)[0];
	node *root=alloc_node(x,v);
	for(i=1;i<n;i++){
		x=(*(v+i)->r)[0];
		insert_node(&root,x,v+i);
	}
	return root;
}
node *build_kd_tree(vertex *v,unsigned int n){
	unsigned int i;
	double x=(*(v)->r)[0];
	int depth=0;
	node *root=alloc_kd_node(x,v,0);
	for(i=1;i<n;i++){
		depth=0;
		x=(*(v+i)->r)[1];
		insert_kd_node(&root,x,v+i,&depth);
	}
	return root;
}
int flip_all_edge_ne(edge *e,unsigned int ne){
	unsigned int i;
	for(i=0;i<ne;i++){
		flip_all_edge(e,ne);
	}
	return 0;
}
int main(int argc,char *argv[]){
	unsigned int n=300,nt=2,ne=5;
	int dsfmt_seed=10013;
	__m128d box={32.0,32.0};
	dsfmt_init_gen_rand(&dsfmt,dsfmt_seed);
	particle *p=particle_alloc(n);
	vertex *v=vertex_alloc(p,n);
	edge *e=edge_alloc(n*4);
	triangle *t=triangle_alloc(n*4);

	init_particle(p,n,box);
	init_square(t,e,v,box);

	build_kd_triangle(t);
	tesselate_walk_kd2(t,e,v,&nt,&ne,n-2);
	//tesselate(t,e,v,&nt,&ne,n-1);
	dump_all_edge(e,ne);
	return 0;
}
