<template>
  <div>
    <v-toolbar light>
      <v-toolbar-title>
        Manage Annotations
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn color="primary" to="/main/annotations/create">Create Annotation</v-btn>
    </v-toolbar>
    <v-data-table :headers="headers" :items="annotations" item-key="name">
      <template v-slot:item="{ item }">
        <tr>
          <td>{{ item.title }}</td>
          <td>
            <v-btn text :to="{name: 'main-annotations-edit', params: {id: item.id}}">
              <v-icon>edit</v-icon>
            </v-btn>
            <v-btn text @click="deleteAnnotation(item.id)">
              <v-icon>delete</v-icon>
            </v-btn>
          </td>
        </tr>
      </template>
    </v-data-table>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Store } from 'vuex';
import { Annotation } from '@/api';
import { readAnnotations } from '@/store/annotation/getters';
import { dispatchGetAnnotations, dispatchDeleteAnnotation } from '@/store/annotation/actions';

@Component
export default class AnnotationAnnotations extends Vue {
  public headers = [
    {
      text: 'Title',
      sortable: true,
      value: 'title',
      align: 'left',
    },
    {
      text: 'Actions',
      value: 'id',
    },
  ];
  get annotations() {
    return readAnnotations(this.$store);
  }

  public async mounted() {
    await dispatchGetAnnotations(this.$store);
  }

  public async deleteAnnotation(id: number) {
    await dispatchDeleteAnnotation(this.$store, {id});
  }
}
</script>
