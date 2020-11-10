<template>
  <div>
    <v-toolbar light>
      <v-toolbar-title>
        Manage Segmentations
      </v-toolbar-title>
      <v-spacer></v-spacer>
      <v-btn color="primary" to="/main/segmentations/create">Create Segmentation</v-btn>
    </v-toolbar>
    <v-data-table :headers="headers" :items="segmentations" item-key="name">
      <template v-slot:item="{ item }">
        <tr>
          <td>{{ item.title }}</td>
          <td>
            <v-btn text :to="{name: 'main-segmentations-edit', params: {id: item.id}}">
              <v-icon>edit</v-icon>
            </v-btn>
            <v-btn text @click="deleteSegmentation(item.id)">
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
import { Segmentation } from '@/interfaces';
import { readSegmentations } from '@/store/segmentation/getters';
import { dispatchGetSegmentations, dispatchDeleteSegmentation } from '@/store/segmentation/actions';

@Component
export default class SegmentationSegmentations extends Vue {
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
  get segmentations() {
    return readSegmentations(this.$store);
  }

  public async mounted() {
    await dispatchGetSegmentations(this.$store);
  }

  public async deleteSegmentation(id: number) {
    await dispatchDeleteSegmentation(this.$store, {id});
  }
}
</script>
