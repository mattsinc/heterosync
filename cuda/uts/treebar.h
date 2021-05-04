/*****************************************************************************/
/* This file is part of the RSIM Applications Library.                       */
/*                                                                           */
/************************ LICENSE TERMS AND CONDITIONS ***********************/
/*                                                                           */
/*  Copyright Notice                                                         */
/*       1997 Rice University                                                */
/*                                                                           */
/*  1. The "Software", below, refers to RSIM (Rice Simulator for ILP         */
/*  Multiprocessors) version 1.0 and includes the RSIM Simulator, the        */
/*  RSIM Applications Library, Example Applications ported to RSIM,          */
/*  and RSIM Utilities.  Each licensee is addressed as "you" or              */
/*  "Licensee."                                                              */
/*                                                                           */
/*  2. Rice University is copyright holder for the RSIM Simulator and RSIM   */
/*  Utilities. The copyright holders reserve all rights except those         */
/*  expressly granted to the Licensee herein.                                */
/*                                                                           */
/*  3. Permission to use, copy, and modify the RSIM Simulator and RSIM       */
/*  Utilities for any non-commercial purpose and without fee is hereby       */
/*  granted provided that the above copyright notice appears in all copies   */
/*  (verbatim or modified) and that both that copyright notice and this      */
/*  permission notice appear in supporting documentation. All other uses,    */
/*  including redistribution in whole or in part, are forbidden without      */
/*  prior written permission.                                                */
/*                                                                           */
/*  4. The RSIM Applications Library is free software; you can               */
/*  redistribute it and/or modify it under the terms of the GNU Library      */
/*  General Public License as published by the Free Software Foundation;     */
/*  either version 2 of the License, or (at your option) any later           */
/*  version.                                                                 */
/*                                                                           */
/*  The Library is distributed in the hope that it will be useful, but       */
/*  WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU         */
/*  Library General Public License for more details.                         */
/*                                                                           */
/*  You should have received a copy of the GNU Library General Public        */
/*  License along with the Library; if not, write to the Free Software       */
/*  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,    */
/*  USA.                                                                     */
/*                                                                           */
/*  5. LICENSEE AGREES THAT THE EXPORT OF GOODS AND/OR TECHNICAL DATA FROM   */
/*  THE UNITED STATES MAY REQUIRE SOME FORM OF EXPORT CONTROL LICENSE FROM   */
/*  THE U.S.  GOVERNMENT AND THAT FAILURE TO OBTAIN SUCH EXPORT CONTROL      */
/*  LICENSE MAY RESULT IN CRIMINAL LIABILITY UNDER U.S. LAWS.                */
/*                                                                           */
/*  6. RICE UNIVERSITY NOR ANY OF THEIR EMPLOYEES MAKE ANY WARRANTY,         */
/*  EXPRESS OR IMPLIED, OR ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY      */
/*  FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION,        */
/*  APPARATUS, PRODUCT, OR PROCESS DISCLOSED AND COVERED BY A LICENSE        */
/*  GRANTED UNDER THIS LICENSE AGREEMENT, OR REPRESENT THAT ITS USE WOULD    */
/*  NOT INFRINGE PRIVATELY OWNED RIGHTS.                                     */
/*                                                                           */
/*  7. IN NO EVENT WILL RICE UNIVERSITY BE LIABLE FOR ANY DAMAGES,           */
/*  INCLUDING DIRECT, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES          */
/*  RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE      */
/*  LICENSED SOFTWARE.                                                       */
/*                                                                           */
/*****************************************************************************/

#ifndef _treebar_h_
#define _treebar_h_ 1

#include <stdlib.h>
#include <string.h>

struct TreeBarNode
{
  int pad1[32];
  volatile int flag[2],which;
  int pad2[32];
};

typedef struct TreeBar
{
  int n;
  struct TreeBarNode *arr;
} TreeBar;

void TreeBarInit(TreeBar *, int);
void TreeBarrier(TreeBar *, int);

void TreeBarInit(TreeBar *x, int n)
{
  x->n=n; /* must be a power of 2 */
  x->arr=(struct TreeBarNode *) malloc(n*sizeof(struct TreeBarNode));
  memset(x->arr,0, n*sizeof(struct TreeBarNode));
  //for (int i=0; i< n; i++)
  //  {
  //    AssociateAddrNode(x->arr+i,x->arr+i+1,i,"bar");
  //  }
}

void TreeBarrier(TreeBar *x, int whoami)
{
  int other,step, which;

  which = x->arr[whoami].which;
  
  for (step=1; step < x->n; step <<=1)
    {
      if (whoami & step)
	{
	  //REL_MEMBAR;
	  __asm__("mfence");

	  x->arr[whoami].flag[which]=1;
	  while (x->arr[whoami].flag[which]);
	  break;
	}
      other=whoami | step;
      while (!x->arr[other].flag[which]);
    }
  
  //ACQ_MEMBAR;
  __asm__("mfence");

  step >>=1;
  
  //REL_MEMBAR;
  __asm__("mfence");

  while (step > 0)
    {
      other=whoami | step;
      x->arr[other].flag[which]=0;
      step >>=1;
    }
  x->arr[whoami].which = 1-which; /* toggle this between 0 and 1 */
}


#endif
